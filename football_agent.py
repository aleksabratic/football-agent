"""
============================================================
 AGENT IA — PRÉDICTIONS FOOTBALLISTIQUES + TELEGRAM
 (SportAPI7 via RapidAPI + The Odds API + modèle Poisson)
============================================================

Cet agent :
  1. Récupère les matchs du jour via SportAPI7
  2. Analyse la forme des équipes + H2H + cotes bookmakers
  3. Prédit le score exact via modèle de Poisson
  4. Cross-référence les cotes avec The Odds API
  5. Génère des prédictions via Claude et envoie sur Telegram

Pré-requis :
    pip install anthropic requests python-dotenv

Variables d'environnement (fichier .env) :
    ANTHROPIC_API_KEY     → console.anthropic.com
    FOOTBALL_API_KEY      → clé RapidAPI (SportAPI7)
    ODDS_API_KEY          → the-odds-api.com (plan gratuit = 500 req/mois)
    TELEGRAM_BOT_TOKEN    → @BotFather sur Telegram
    TELEGRAM_CHAT_ID      → ton ID de chat

Lancement :
    py football_agent.py
============================================================
"""

import sys

sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")

import os
import json
import math
import time
import logging
import requests
from datetime import datetime, date, timezone, timedelta
from dotenv import load_dotenv
import anthropic

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)

# ─────────────────────────────────────────────
#  CONFIGURATION
# ─────────────────────────────────────────────
load_dotenv()

ANTHROPIC_API_KEY  = os.environ["ANTHROPIC_API_KEY"]
FOOTBALL_API_KEY   = os.environ["FOOTBALL_API_KEY"]
ODDS_API_KEY       = os.environ.get("ODDS_API_KEY", "")
TELEGRAM_BOT_TOKEN = os.environ["TELEGRAM_BOT_TOKEN"]
TELEGRAM_CHAT_ID   = os.environ["TELEGRAM_CHAT_ID"]

SPORT_API_BASE = "https://sportapi7.p.rapidapi.com/api/v1"
SPORT_API_HEADERS = {
    "x-rapidapi-host": "sportapi7.p.rapidapi.com",
    "x-rapidapi-key":  FOOTBALL_API_KEY,
    "Content-Type":    "application/json",
}

ODDS_API_BASE = "https://api.the-odds-api.com/v4"

# Slugs uniqueTournament à suivre → nom affiché
TARGET_SLUGS: dict[str, str] = {
    "world-championship": "FIFA Coupe du Monde 2026",
    "premier-league":     "Premier League",
    "laliga":             "La Liga",
    "bundesliga":         "Bundesliga",
    "serie-a":            "Serie A",
    "ligue-1":            "Ligue 1",
}

# Tournois joués sur terrain neutre (pas d'avantage domicile)
NEUTRAL_VENUE_SLUGS: set[str] = {"world-championship"}

# Mapping slug → clé sport The Odds API
ODDS_SPORT_MAP: dict[str, str] = {
    "world-championship": "soccer_fifa_world_cup",
    "premier-league":     "soccer_epl",
    "laliga":             "soccer_spain_la_liga",
    "bundesliga":         "soccer_germany_bundesliga",
    "serie-a":            "soccer_italy_serie_a",
    "ligue-1":            "soccer_france_ligue_1",
}

# Traductions FR→EN pour correspondance noms d'équipes (The Odds API = anglais)
FR_TO_EN_TEAMS: dict[str, str] = {
    "espagne": "spain", "france": "france", "allemagne": "germany",
    "angleterre": "england", "italie": "italy", "brésil": "brazil",
    "argentine": "argentina", "croatie": "croatia", "maroc": "morocco",
    "tunisie": "tunisia", "sénégal": "senegal", "côte d'ivoire": "ivory coast",
    "états-unis": "usa", "etats-unis": "usa", "mexique": "mexico",
    "japon": "japan", "corée du sud": "south korea", "australie": "australia",
    "pays-bas": "netherlands", "belgique": "belgium", "suisse": "switzerland",
    "pologne": "poland", "suède": "sweden", "danemark": "denmark",
    "serbie": "serbia", "ukraine": "ukraine", "portugal": "portugal",
    "ghana": "ghana", "nigeria": "nigeria", "cameroun": "cameroon",
    "algérie": "algeria", "équateur": "ecuador", "colombie": "colombia",
    "chili": "chile", "pérou": "peru", "uruguay": "uruguay",
    "canada": "canada", "costa rica": "costa rica", "panama": "panama",
    "arabie saoudite": "saudi arabia", "iran": "iran",
    "nouvelle-zélande": "new zealand", "venezuela": "venezuela",
}

MODEL  = "claude-opus-4-8"
client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)


# ─────────────────────────────────────────────
#  HELPERS INTERNES
# ─────────────────────────────────────────────

def _sport_get(endpoint: str, params: dict = None, _retries: int = 4) -> dict:
    """Appel GET vers SportAPI7 avec retry exponentiel sur 429."""
    url = f"{SPORT_API_BASE}/{endpoint}"
    delay = 5
    for attempt in range(_retries):
        resp = requests.get(url, headers=SPORT_API_HEADERS, params=params or {}, timeout=15)
        if resp.status_code == 429:
            if attempt < _retries - 1:
                print(f"[SportAPI7] 429 rate limit — attente {delay}s avant retry {attempt + 2}/{_retries}...")
                time.sleep(delay)
                delay *= 2
                continue
        resp.raise_for_status()
        return resp.json()
    resp.raise_for_status()
    return {}


def _fractional_to_decimal(frac: str) -> float | None:
    """Convertit une cote fractionnaire '3/2' en décimale 2.5."""
    try:
        if "/" in frac:
            n, d = frac.split("/")
            return round(int(n) / int(d) + 1, 2)
        return round(float(frac), 2)
    except (ValueError, ZeroDivisionError):
        return None


_TZ_FR = timezone(timedelta(hours=2))

def _ts_to_fr(ts: int) -> str:
    """Convertit un timestamp Unix en heure française (UTC+2 été)."""
    return datetime.fromtimestamp(ts, tz=_TZ_FR).strftime("%H:%M")


def _poisson_prob(lam: float, k: int) -> float:
    """Probabilité de Poisson P(X=k) pour une moyenne λ."""
    if lam <= 0:
        return 1.0 if k == 0 else 0.0
    return math.exp(-lam) * (lam ** k) / math.factorial(k)


def _to_en_team(fr_name: str) -> str:
    """Traduit un nom d'équipe FR→EN pour The Odds API."""
    return FR_TO_EN_TEAMS.get(fr_name.lower().strip(), fr_name.lower().strip())


# ─────────────────────────────────────────────
#  OUTILS
# ─────────────────────────────────────────────

def get_fixtures_today() -> str:
    """
    Récupère tous les matchs du jour pour les ligues cibles.
    Retourne event_id, IDs des équipes, heure locale, statut, terrain neutre.
    """
    today = date.today().isoformat()
    data  = _sport_get(f"sport/football/scheduled-events/{today}")

    result: list[dict] = []
    for e in data.get("events", []):
        ut   = e.get("tournament", {}).get("uniqueTournament", {})
        slug = ut.get("slug", "")
        if slug not in TARGET_SLUGS:
            continue
        result.append({
            "event_id":         e["id"],
            "tournament":       TARGET_SLUGS[slug],
            "slug":             slug,
            "is_neutral_venue": slug in NEUTRAL_VENUE_SLUGS,
            "heure_fr":         _ts_to_fr(e["startTimestamp"]) if e.get("startTimestamp") else "?",
            "home_team":        e["homeTeam"]["name"],
            "home_id":          e["homeTeam"]["id"],
            "away_team":        e["awayTeam"]["name"],
            "away_id":          e["awayTeam"]["id"],
            "statut":           e.get("status", {}).get("description", "?"),
        })

    if not result:
        return "Aucun match trouvé aujourd'hui pour les ligues sélectionnées."
    return json.dumps(result, ensure_ascii=False, indent=2)


def get_team_form(team_id: int) -> str:
    """
    Retourne les 5 derniers matchs d'une équipe.
    Inclut : W/D/L, buts marqués/encaissés, adversaire, compétition.
    Les champs buts_marqués_5j et buts_encaissés_5j servent au modèle Poisson.
    """
    data   = _sport_get(f"team/{team_id}/events/last/0")
    events = data.get("events", [])[-5:]

    recent: list[dict] = []
    for e in events:
        ht  = e["homeTeam"]
        at  = e["awayTeam"]
        hs  = e.get("homeScore", {}).get("current") or 0
        as_ = e.get("awayScore", {}).get("current") or 0
        is_home = ht["id"] == team_id
        gf = hs if is_home else as_
        ga = as_ if is_home else hs

        if is_home:
            res = "V" if hs > as_ else ("N" if hs == as_ else "D")
        else:
            res = "V" if as_ > hs else ("N" if hs == as_ else "D")

        recent.append({
            "date":       datetime.fromtimestamp(e["startTimestamp"]).strftime("%Y-%m-%d"),
            "adversaire": at["name"] if is_home else ht["name"],
            "domicile":   is_home,
            "score":      f"{gf}-{ga}",
            "résultat":   res,
            "tournoi":    e.get("tournament", {}).get("name", "?"),
        })

    wins  = sum(1 for m in recent if m["résultat"] == "V")
    draws = sum(1 for m in recent if m["résultat"] == "N")
    loss  = sum(1 for m in recent if m["résultat"] == "D")
    gf_total = sum(int(m["score"].split("-")[0]) for m in recent)
    ga_total = sum(int(m["score"].split("-")[1]) for m in recent)

    return json.dumps({
        "bilan_5j":           f"{wins}V-{draws}N-{loss}D",
        "buts_marqués_5j":    gf_total,
        "buts_encaissés_5j":  ga_total,
        "matchs":             recent,
    }, ensure_ascii=False, indent=2)


def get_head_to_head(event_id: int) -> str:
    """
    Retourne le bilan des confrontations directes entre les deux équipes.
    Note : en Coupe du Monde, ces stats reflètent des matchs passés avec
    domicile/extérieur éventuellement différents, pas la Coupe du Monde actuelle.
    """
    data = _sport_get(f"event/{event_id}/h2h")
    duel = data.get("teamDuel") or {}
    if not duel:
        return "Pas d'historique H2H disponible pour ce match (première rencontre ou données manquantes)."
    return json.dumps({
        "victoires_équipe1": duel.get("homeWins", 0),
        "nuls":              duel.get("draws",    0),
        "victoires_équipe2": duel.get("awayWins", 0),
    }, ensure_ascii=False, indent=2)


def get_event_odds(event_id: int) -> str:
    """
    Retourne les cotes 1X2 et Over/Under 2.5 buts pour un match (SportAPI7).
    Cotes converties en format décimal.
    """
    time.sleep(0.4)  # Évite le rate limiting 429
    data = _sport_get(f"event/{event_id}/odds/1/all")

    result: dict = {}
    for market in data.get("markets", []):
        group = market.get("marketGroup", "")
        name  = market.get("marketName", "")

        if group == "1X2" and "Full time" in name:
            for c in market.get("choices", []):
                dec = _fractional_to_decimal(c.get("fractionalValue", ""))
                result[f"cote_{c['name']}"] = dec

        elif "Total" in name or "Over/Under" in name:
            for c in market.get("choices", []):
                handicap = str(c.get("handicap", c.get("point", "")))
                if "2.5" in handicap:
                    label = f"cote_{c['name'].lower().replace(' ', '_')}_2.5"
                    dec = _fractional_to_decimal(c.get("fractionalValue", ""))
                    result[label] = dec

    if not result:
        return "Cotes non disponibles pour ce match."
    return json.dumps(result, ensure_ascii=False, indent=2)


def predict_score(
    home_goals_scored_5: int,
    home_goals_conceded_5: int,
    away_goals_scored_5: int,
    away_goals_conceded_5: int,
    is_neutral_venue: bool = False,
) -> str:
    """
    Prédit le score exact le plus probable via le modèle de Poisson.
    Paramètres extraits de get_team_form (buts_marqués_5j / buts_encaissés_5j).
    Pour la Coupe du Monde (terrain neutre), aucun avantage domicile n'est appliqué.
    """
    n = 5
    avg_hgs = home_goals_scored_5 / n    # buts/match équipe 1 (attaque)
    avg_hgc = home_goals_conceded_5 / n  # buts encaissés/match équipe 1 (défense)
    avg_ags = away_goals_scored_5 / n    # buts/match équipe 2 (attaque)
    avg_agc = away_goals_conceded_5 / n  # buts encaissés/match équipe 2 (défense)

    # Buts attendus : combinaison attaque de l'un / défense de l'autre
    lam_1 = (avg_hgs + avg_agc) / 2
    lam_2 = (avg_ags + avg_hgc) / 2

    # Avantage terrain uniquement pour les ligues (pas en WC)
    if not is_neutral_venue:
        lam_1 *= 1.15

    # Probabilités Poisson pour scores de 0 à 5 buts par équipe
    scores = []
    for g1 in range(6):
        for g2 in range(6):
            p = _poisson_prob(lam_1, g1) * _poisson_prob(lam_2, g2) * 100
            scores.append((g1, g2, round(p, 2)))

    scores.sort(key=lambda x: x[2], reverse=True)
    top5 = scores[:5]

    # Probabilités de résultat global
    p_eq1_win = sum(p for g1, g2, p in scores if g1 > g2)
    p_draw    = sum(p for g1, g2, p in scores if g1 == g2)
    p_eq2_win = sum(p for g1, g2, p in scores if g2 > g1)

    return json.dumps({
        "buts_attendus_équipe1":       round(lam_1, 2),
        "buts_attendus_équipe2":       round(lam_2, 2),
        "terrain_neutre":              is_neutral_venue,
        "top_scores_probables":        [
            {"score": f"{g1}-{g2}", "probabilité_%": p}
            for g1, g2, p in top5
        ],
        "probabilité_victoire_éq1_%":  round(p_eq1_win, 1),
        "probabilité_nul_%":           round(p_draw,    1),
        "probabilité_victoire_éq2_%":  round(p_eq2_win, 1),
    }, ensure_ascii=False, indent=2)


def get_odds_api_h2h(home_team: str, away_team: str, slug: str) -> str:
    """
    Récupère les cotes 1X2 depuis The Odds API (cross-référence bookmakers EU).
    Utile quand SportAPI7 retourne 429 ou pour vérifier les cotes sur plusieurs bookmakers.
    """
    if not ODDS_API_KEY:
        return "ODDS_API_KEY non configurée."

    sport_key = ODDS_SPORT_MAP.get(slug)
    if not sport_key:
        return f"Compétition '{slug}' non supportée par The Odds API."

    try:
        resp = requests.get(
            f"{ODDS_API_BASE}/sports/{sport_key}/odds",
            params={
                "apiKey":     ODDS_API_KEY,
                "regions":    "eu",
                "markets":    "h2h",
                "oddsFormat": "decimal",
                "dateFormat": "iso",
            },
            timeout=15,
        )
        if resp.status_code == 404:
            return f"Compétition '{sport_key}' non disponible sur The Odds API (pas encore démarrée ?)."
        resp.raise_for_status()
        events_list = resp.json()
    except Exception as e:
        return f"Erreur The Odds API : {e}"

    # Correspondance par nom d'équipe (FR → EN)
    home_en = _to_en_team(home_team)
    away_en = _to_en_team(away_team)

    matched = None
    for ev in events_list:
        ev_home = ev.get("home_team", "").lower()
        ev_away = ev.get("away_team", "").lower()
        home_ok = home_en in ev_home or ev_home in home_en or home_en[:4] == ev_home[:4]
        away_ok = away_en in ev_away or ev_away in away_en or away_en[:4] == ev_away[:4]
        if home_ok and away_ok:
            matched = ev
            break

    if not matched:
        return f"Match '{home_team} vs {away_team}' introuvable sur The Odds API."

    # Trouver le bookmaker avec le plus de marchés
    best_book = max(matched.get("bookmakers", []), key=lambda b: len(b.get("markets", [])), default=None)
    if not best_book:
        return "Aucun bookmaker disponible pour ce match sur The Odds API."

    result = {"bookmaker": best_book.get("title", "?")}
    for market in best_book.get("markets", []):
        if market["key"] == "h2h":
            for outcome in market.get("outcomes", []):
                result[f"cote_{outcome['name']}"] = round(outcome["price"], 2)

    remaining = resp.headers.get("x-requests-remaining", "?")
    result["requêtes_restantes_odds_api"] = remaining

    return json.dumps(result, ensure_ascii=False, indent=2)


def send_telegram_report(text: str) -> str:
    """
    Envoie un message formaté HTML sur Telegram.
    Gère la limite de 4096 caractères par message.
    """
    url    = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    chunks = [text[i:i + 4000] for i in range(0, len(text), 4000)]
    for chunk in chunks:
        payload = {"chat_id": TELEGRAM_CHAT_ID, "text": chunk, "parse_mode": "HTML"}
        resp    = requests.post(url, json=payload, timeout=15)
        resp.raise_for_status()
    return f"✅ Message envoyé sur Telegram ({len(chunks)} partie(s))."


# ─────────────────────────────────────────────
#  DESCRIPTIONS DES OUTILS
# ─────────────────────────────────────────────

TOOLS = [
    {
        "name": "get_fixtures_today",
        "description": (
            "Récupère tous les matchs du jour pour : FIFA Coupe du Monde, Premier League, "
            "La Liga, Bundesliga, Serie A, Ligue 1. "
            "Retourne event_id, home/away team + IDs, heure française, statut, is_neutral_venue. "
            "Toujours appeler EN PREMIER, sans argument."
        ),
        "input_schema": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "get_team_form",
        "description": (
            "Retourne les 5 derniers matchs d'une équipe avec bilan W/D/L et buts. "
            "Les champs buts_marqués_5j et buts_encaissés_5j sont utilisés ensuite par predict_score. "
            "Appeler pour l'équipe 1 (home_id) ET l'équipe 2 (away_id) de chaque match."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "team_id": {
                    "type": "integer",
                    "description": "ID de l'équipe (home_id ou away_id fourni par get_fixtures_today)",
                },
            },
            "required": ["team_id"],
        },
    },
    {
        "name": "get_head_to_head",
        "description": (
            "Retourne le bilan historique H2H entre les deux équipes. "
            "Les champs sont victoires_équipe1 / nuls / victoires_équipe2 (pas domicile/extérieur). "
            "Utiliser event_id fourni par get_fixtures_today."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "event_id": {
                    "type": "integer",
                    "description": "ID du match (fourni par get_fixtures_today)",
                },
            },
            "required": ["event_id"],
        },
    },
    {
        "name": "get_event_odds",
        "description": (
            "Retourne les cotes 1X2 et Over/Under 2.5 buts en format décimal (SportAPI7). "
            "Appeler pour chaque match avant de générer la prédiction."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "event_id": {
                    "type": "integer",
                    "description": "ID du match (fourni par get_fixtures_today)",
                },
            },
            "required": ["event_id"],
        },
    },
    {
        "name": "predict_score",
        "description": (
            "Calcule le score exact le plus probable via le modèle statistique de Poisson. "
            "Utilise les buts marqués/encaissés sur 5 matchs récupérés par get_team_form. "
            "Pour la Coupe du Monde, mettre is_neutral_venue=true (pas d'avantage domicile). "
            "Appeler après get_team_form pour les deux équipes de chaque match."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "home_goals_scored_5": {
                    "type": "integer",
                    "description": "buts_marqués_5j de l'équipe 1 (home team)",
                },
                "home_goals_conceded_5": {
                    "type": "integer",
                    "description": "buts_encaissés_5j de l'équipe 1 (home team)",
                },
                "away_goals_scored_5": {
                    "type": "integer",
                    "description": "buts_marqués_5j de l'équipe 2 (away team)",
                },
                "away_goals_conceded_5": {
                    "type": "integer",
                    "description": "buts_encaissés_5j de l'équipe 2 (away team)",
                },
                "is_neutral_venue": {
                    "type": "boolean",
                    "description": "true pour la Coupe du Monde (terrain neutre), false pour les ligues",
                },
            },
            "required": [
                "home_goals_scored_5",
                "home_goals_conceded_5",
                "away_goals_scored_5",
                "away_goals_conceded_5",
                "is_neutral_venue",
            ],
        },
    },
    {
        "name": "get_odds_api_h2h",
        "description": (
            "Récupère les cotes 1X2 depuis The Odds API (bookmakers européens). "
            "Utiliser quand get_event_odds retourne 'non disponibles' (429 SportAPI7). "
            "Passer home_team et away_team exacts de get_fixtures_today, et le slug du tournoi."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "home_team": {
                    "type": "string",
                    "description": "Nom exact de l'équipe domicile (depuis get_fixtures_today)",
                },
                "away_team": {
                    "type": "string",
                    "description": "Nom exact de l'équipe extérieure (depuis get_fixtures_today)",
                },
                "slug": {
                    "type": "string",
                    "description": "Slug du tournoi (depuis get_fixtures_today, ex: 'world-championship')",
                },
            },
            "required": ["home_team", "away_team", "slug"],
        },
    },
    {
        "name": "send_telegram_report",
        "description": (
            "Envoie le rapport final d'analyse sur Telegram. "
            "Supporte HTML : <b>gras</b>, <i>italique</i>. "
            "Appeler UNE SEULE FOIS à la toute fin avec l'intégralité du rapport."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "text": {
                    "type": "string",
                    "description": "Rapport complet formaté en HTML pour Telegram",
                },
            },
            "required": ["text"],
        },
    },
]


# ─────────────────────────────────────────────
#  RÉPARTITEUR D'OUTILS
# ─────────────────────────────────────────────

def execute_tool(name: str, tool_input: dict) -> str:
    try:
        match name:
            case "get_fixtures_today":
                return get_fixtures_today()
            case "get_team_form":
                return get_team_form(tool_input["team_id"])
            case "get_head_to_head":
                return get_head_to_head(tool_input["event_id"])
            case "get_event_odds":
                return get_event_odds(tool_input["event_id"])
            case "predict_score":
                return predict_score(
                    home_goals_scored_5   = tool_input["home_goals_scored_5"],
                    home_goals_conceded_5 = tool_input["home_goals_conceded_5"],
                    away_goals_scored_5   = tool_input["away_goals_scored_5"],
                    away_goals_conceded_5 = tool_input["away_goals_conceded_5"],
                    is_neutral_venue      = tool_input.get("is_neutral_venue", False),
                )
            case "get_odds_api_h2h":
                return get_odds_api_h2h(
                    home_team = tool_input["home_team"],
                    away_team = tool_input["away_team"],
                    slug      = tool_input["slug"],
                )
            case "send_telegram_report":
                return send_telegram_report(tool_input["text"])
            case _:
                return f"Outil inconnu : {name}"
    except Exception as e:
        return f"[ERREUR outil {name}] {e}"


# ─────────────────────────────────────────────
#  PROMPT SYSTÈME
# ─────────────────────────────────────────────

def _build_system_prompt() -> str:
    today = date.today().strftime('%d/%m/%Y')
    return f"""Tu es un expert en analyse footballistique et statistiques. Ta mission du {today} :

ÉTAPES OBLIGATOIRES (dans cet ordre) :
1. Appelle get_fixtures_today (sans argument).
2. S'il n'y a aucun match → envoie un message Telegram le signalant et arrête-toi.
3. Sélection des matchs :
   - FIFA Coupe du Monde : TOUS les matchs du jour sans exception.
   - Autres ligues : max 3 matchs par ligue, 6 au total.
4. Pour chaque match sélectionné :
   a. get_team_form(home_id)   → note buts_marqués_5j et buts_encaissés_5j
   b. get_team_form(away_id)   → note buts_marqués_5j et buts_encaissés_5j
   c. get_head_to_head(event_id)
   d. get_event_odds(event_id) → si retourne "non disponibles", appelle get_odds_api_h2h(home_team, away_team, slug)
   e. predict_score(home_goals_scored_5, home_goals_conceded_5, away_goals_scored_5, away_goals_conceded_5, is_neutral_venue)
5. Génère le rapport et appelle send_telegram_report UNE SEULE FOIS.

RÈGLE COUPE DU MONDE (CRITIQUE) :
- is_neutral_venue = true pour tous les matchs "world-championship".
- Les deux équipes jouent sur terrain neutre (stades américains/canadiens/mexicains).
- NE JAMAIS utiliser les termes "domicile" ou "extérieur" pour qualifier les performances passées dans le contexte WC. Utilise les noms des équipes directement.
- En H2H, les champs victoires_équipe1 / victoires_équipe2 correspondent à l'équipe listée en premier/second, pas à un avantage terrain.
- En forme récente, les colonnes "Dom/Ext" concernent les matchs PASSÉS de l'équipe, pas la WC actuelle.

FORMAT DU RAPPORT TELEGRAM (HTML strict) :

<b>⚽ PRONOSTICS DU {date.today().strftime('%d/%m/%Y')}</b>
<i>Analyse : forme récente · H2H · cotes · modèle Poisson</i>

━━━━━━━━━━━━━━━━━━━━━━━━

Si matchs WC présents :
<b>🌍 FIFA COUPE DU MONDE 2026</b>

Puis matchs de clubs :
<b>🏆 LIGUES EUROPÉENNES</b>

Pour chaque match :
<b>🏟 [Équipe A] vs [Équipe B]</b>
<i>[Compétition] — 🕐 [Heure]</i>

📊 <b>Forme (5 derniers matchs) :</b>
  [Éq. A] : [résultats] — [X] buts marqués / [X] encaissés
  [Éq. B] : [résultats] — [X] buts marqués / [X] encaissés

⚖️ <b>H2H :</b> [X] victoires [Éq. A] / [X] nuls / [X] victoires [Éq. B]

💹 <b>Cotes :</b> [Éq. A] [X.XX] | Nul [X.XX] | [Éq. B] [X.XX] | O2.5 [X.XX]

🎲 <b>Score prédit (Poisson) :</b> [X-X] ([Y]%) — alt: [X-X] ([Y]%)

🎯 <b>Pronostic :</b> [ta prédiction]
📝 <b>Raison :</b> [justification factuelle en 1-2 phrases]
⭐ <b>Confiance :</b> [🔴 Faible | 🟡 Moyenne | 🟢 Élevée]

━━━━━━━━━━━━━━━━━━━━━━━━

<b>🔥 TOP PICKS DU JOUR</b>
1. [Match] — [Pronostic] ⭐⭐⭐
2. [Match] — [Pronostic] ⭐⭐
3. [Match] — [Pronostic] ⭐⭐

<i>⚠️ Ces analyses sont informatives. Pariez de manière responsable.</i>

RÈGLES GÉNÉRALES :
- heure_fr dans les données est déjà en UTC+2 (heure française été)
- Sois factuel et concis
- Pour les cotes : utilise les noms réels des équipes (pas "1" ou "2")
- Si les cotes restent indisponibles malgré les deux tentatives, indique-le brièvement
- Le score Poisson est une estimation statistique, mentionne-le comme tel
"""


# ─────────────────────────────────────────────
#  BOUCLE DE L'AGENT
# ─────────────────────────────────────────────

def run_agent(max_steps: int = 80) -> None:
    today_str     = date.today().strftime('%d/%m/%Y')
    system_prompt = _build_system_prompt()

    messages = [{
        "role": "user",
        "content": f"Analyse les matchs de football du {today_str} et envoie le rapport complet sur Telegram.",
    }]

    logging.info("=" * 55)
    logging.info(f"AGENT FOOT ⚽ — {today_str}")
    logging.info("=" * 55)

    for step in range(max_steps):
        logging.info(f"--- Tour {step + 1} ---")

        response = client.messages.create(
            model=MODEL,
            max_tokens=8192,
            system=system_prompt,
            tools=TOOLS,
            messages=messages,
        )

        for block in response.content:
            if block.type == "text" and block.text.strip():
                logging.info(f"[Claude] {block.text}")

        if response.stop_reason != "tool_use":
            logging.info("✅ Agent terminé avec succès.")
            return

        messages.append({"role": "assistant", "content": response.content})

        tool_results = []
        for block in response.content:
            if block.type == "tool_use":
                logging.info(f"[Outil] {block.name}  input={json.dumps(block.input, ensure_ascii=False)}")
                result  = execute_tool(block.name, block.input)
                preview = result[:300] + " [...]" if len(result) > 300 else result
                logging.info(f"[Résultat] {preview}")
                tool_results.append({
                    "type":        "tool_result",
                    "tool_use_id": block.id,
                    "content":     result,
                })

        messages.append({"role": "user", "content": tool_results})

    logging.warning("⚠️ Nombre maximum d'étapes atteint sans réponse finale.")


# ─────────────────────────────────────────────
#  LANCEMENT
# ─────────────────────────────────────────────

if __name__ == "__main__":
    run_agent()

"""
============================================================
 AGENT IA — PRONOSTICS FOOT (VALUE BETTING)
 (API-Football api-sports.io — plan Free 100 req/jour)
============================================================

Architecture de décision :
  - Moteur probabiliste en Python (Poisson forces relatives + shrinkage)
  - Dévigging des cotes (marge bookmaker retirée)
  - Pick uniquement si edge > 3% ET EV > 4% (sinon "Aucune valeur")
  - Mise = Kelly/4 (proportionnel à l'avantage détecté)
  - Claude s'occupe uniquement du formatage du rapport

Variables d'environnement (.env) :
    ANTHROPIC_API_KEY   → console.anthropic.com
    APIFOOTBALL_KEY     → api-sports.io (plan Free = 100 req/jour)
    TELEGRAM_BOT_TOKEN  → @BotFather sur Telegram
    TELEGRAM_CHAT_ID    → ton ID de chat
============================================================
"""

import sys

sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")

import csv
import os
import json
import math
import time
import logging
import requests
from datetime import datetime, date, timezone, timedelta
from pathlib import Path
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
APIFOOTBALL_KEY    = os.environ["APIFOOTBALL_KEY"]
TELEGRAM_BOT_TOKEN = os.environ["TELEGRAM_BOT_TOKEN"]
TELEGRAM_CHAT_ID   = os.environ["TELEGRAM_CHAT_ID"]

APIFOOTBALL_BASE    = "https://v3.football.api-sports.io"
APIFOOTBALL_HEADERS = {"x-apisports-key": APIFOOTBALL_KEY}

TARGET_LEAGUES: dict[int, str] = {
    1:   "FIFA Coupe du Monde 2026",
    39:  "Premier League",
    140: "La Liga",
    78:  "Bundesliga",
    135: "Serie A",
    61:  "Ligue 1",
}
NEUTRAL_LEAGUES: set[int] = {1}

# Buts moyens historiques par match dans chaque ligue (domicile, extérieur)
# Utilisés comme prior pour les forces relatives et la régularisation
LEAGUE_GOALS: dict[int, tuple[float, float]] = {
    1:   (1.30, 1.20),  # WC (terrain neutre → moyenne des deux = 1.25)
    39:  (1.55, 1.15),  # Premier League
    140: (1.45, 1.15),  # La Liga
    78:  (1.75, 1.35),  # Bundesliga
    135: (1.35, 1.10),  # Serie A
    61:  (1.40, 1.15),  # Ligue 1
}

# Seuils value betting
EDGE_THRESHOLD = 0.03   # avantage minimum sur le marché (3 %)
EV_THRESHOLD   = 0.04   # espérance de gain minimum (4 %)
KELLY_FRACTION = 0.25   # quart de Kelly pour limiter la variance

PERF_FILE = Path(__file__).parent / "bet_history.csv"
_TZ_FR    = timezone(timedelta(hours=2))
MODEL     = "claude-opus-4-8"
client    = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)


# ─────────────────────────────────────────────
#  HELPER API
# ─────────────────────────────────────────────

def _api_get(endpoint: str, params: dict = None) -> list:
    url   = f"{APIFOOTBALL_BASE}/{endpoint}"
    delay = 5
    for attempt in range(4):
        resp = requests.get(url, headers=APIFOOTBALL_HEADERS, params=params or {}, timeout=15)
        if resp.status_code == 429:
            if attempt < 3:
                logging.warning(f"429 rate limit — attente {delay}s (retry {attempt+2}/4)...")
                time.sleep(delay)
                delay *= 2
                continue
        resp.raise_for_status()
        data = resp.json()
        if data.get("errors"):
            raise RuntimeError(f"API errors: {data['errors']}")
        return data.get("response", [])
    return []


def _ts_to_fr(ts: int) -> str:
    return datetime.fromtimestamp(ts, tz=_TZ_FR).strftime("%H:%M")


# ─────────────────────────────────────────────
#  MOTEUR POISSON (forces relatives + shrinkage)
# ─────────────────────────────────────────────

def _poisson_prob(lam: float, k: int) -> float:
    if lam <= 0:
        return 1.0 if k == 0 else 0.0
    return math.exp(-lam) * (lam ** k) / math.factorial(k)


def _shrink(observed: float, n_games: int, prior: float = 1.0) -> float:
    """
    Régresse la force observée vers le prior (1.0 = moyenne de ligue).
    Avec n_games=3 → 30% données réelles, 70% prior.
    Avec n_games=10+ → 100% données réelles.
    Évite qu'un 2.3 buts/match sur 3 matchs hallucine un grand favori.
    """
    w = min(n_games, 10) / 10.0
    return w * observed + (1.0 - w) * prior


def _relative_poisson(
    h_avg_for: float, h_avg_ag: float,
    a_avg_for: float, a_avg_ag: float,
    h_played: int, a_played: int,
    league_id: int, is_neutral: bool,
) -> tuple[float, float]:
    """
    Calcule les buts attendus (λ) via forces relatives.

    Force d'attaque  = buts marqués / équipe / moyenne de la compétition
    Force de défense = buts encaissés / équipe / moyenne de la compétition
    λ_dom = att_dom × def_ext × moyenne_buts_domicile_compétition
    λ_ext = att_ext × def_dom × moyenne_buts_extérieur_compétition

    Sur terrain neutre (WC) : même base de référence pour les deux équipes.
    """
    avg_h, avg_a = LEAGUE_GOALS.get(league_id, (1.40, 1.15))
    overall_avg  = (avg_h + avg_a) / 2.0

    # Forces brutes (ratio vs moyenne globale de la ligue)
    att_h_raw = h_avg_for / overall_avg if overall_avg > 0 else 1.0
    def_h_raw = h_avg_ag  / overall_avg if overall_avg > 0 else 1.0
    att_a_raw = a_avg_for / overall_avg if overall_avg > 0 else 1.0
    def_a_raw = a_avg_ag  / overall_avg if overall_avg > 0 else 1.0

    # Shrinkage : revenir vers 1.0 proportionnellement au manque de données
    att_h = _shrink(att_h_raw, h_played)
    def_h = _shrink(def_h_raw, h_played)
    att_a = _shrink(att_a_raw, a_played)
    def_a = _shrink(def_a_raw, a_played)

    if is_neutral:
        lam_home = att_h * def_a * overall_avg
        lam_away = att_a * def_h * overall_avg
    else:
        lam_home = att_h * def_a * avg_h * 1.10  # légère prime domicile
        lam_away = att_a * def_h * avg_a

    return round(lam_home, 3), round(lam_away, 3)


def _poisson_1x2(lam_home: float, lam_away: float) -> tuple[float, float, float]:
    """Intègre la matrice des scores Poisson pour obtenir les probas 1X2."""
    p_dom = p_nul = p_ext = 0.0
    for g1 in range(10):
        for g2 in range(10):
            p = _poisson_prob(lam_home, g1) * _poisson_prob(lam_away, g2)
            if   g1 > g2: p_dom += p
            elif g1 == g2: p_nul += p
            else:          p_ext += p
    return p_dom, p_nul, p_ext


def _top_scores(lam_home: float, lam_away: float, n: int = 5) -> list[dict]:
    scores = []
    for g1 in range(7):
        for g2 in range(7):
            prob = _poisson_prob(lam_home, g1) * _poisson_prob(lam_away, g2) * 100
            scores.append({"score": f"{g1}-{g2}", "probabilité_%": round(prob, 2)})
    scores.sort(key=lambda x: x["probabilité_%"], reverse=True)
    return scores[:n]


# ─────────────────────────────────────────────
#  VALUE BETTING : DEVIG + EDGE + EV + KELLY
# ─────────────────────────────────────────────

def _devig(cote_dom: float, cote_nul: float, cote_ext: float) -> dict:
    """
    Retire la marge bookmaker (méthode multiplicative).
    La somme des probas retournées est exactement 1.0.
    """
    total = 1.0 / cote_dom + 1.0 / cote_nul + 1.0 / cote_ext
    return {
        "p_dom":   (1.0 / cote_dom) / total,
        "p_nul":   (1.0 / cote_nul) / total,
        "p_ext":   (1.0 / cote_ext) / total,
        "marge_%": round((total - 1.0) * 100, 2),
    }


def _compute_value_bets(
    p_dom: float, p_nul: float, p_ext: float,
    cote_dom: float, cote_nul: float, cote_ext: float,
) -> list[dict]:
    """
    Identifie les paris à valeur positive.
    Un pari est retenu seulement si :
      edge  = p_modèle − p_marché_juste  > EDGE_THRESHOLD (3 %)
      EV    = p_modèle × cote − 1        > EV_THRESHOLD   (4 %)
    La mise est le quart de Kelly pour limiter la variance.
    """
    fair = _devig(cote_dom, cote_nul, cote_ext)
    bets = []
    for label, p_model, cote, p_fair in [
        ("1 (domicile)", p_dom, cote_dom, fair["p_dom"]),
        ("X (nul)",      p_nul, cote_nul, fair["p_nul"]),
        ("2 (extérieur)", p_ext, cote_ext, fair["p_ext"]),
    ]:
        if not cote or cote <= 1.0:
            continue
        edge  = p_model - p_fair
        ev    = p_model * cote - 1.0
        kelly = max(0.0, ev / (cote - 1.0)) * KELLY_FRACTION
        if edge >= EDGE_THRESHOLD and ev >= EV_THRESHOLD:
            bets.append({
                "issue":            label,
                "p_modèle_%":       round(p_model * 100, 1),
                "p_marché_juste_%": round(p_fair  * 100, 1),
                "edge_%":           round(edge    * 100, 1),
                "EV_%":             round(ev      * 100, 1),
                "cote":             cote,
                "mise_kelly_%":     round(kelly   * 100, 1),
            })
    return sorted(bets, key=lambda x: x["EV_%"], reverse=True)


# ─────────────────────────────────────────────
#  SUIVI DES PERFORMANCES
# ─────────────────────────────────────────────

def _log_bets_to_csv(date_str: str, match_label: str, fixture_id: int, bets: list[dict]) -> None:
    """
    Enregistre chaque value bet dans bet_history.csv.
    Remplis la colonne "résultat" après le match pour calculer ROI et yield.
    CLV (Closing Line Value) peut être ajoutée en comparant cote_prise vs cote_clôture.
    """
    if not bets:
        return
    fieldnames = ["date", "fixture_id", "match", "issue", "cote", "ev_%", "kelly_%", "résultat", "clv"]
    needs_header = not PERF_FILE.exists()
    with open(PERF_FILE, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        if needs_header:
            w.writeheader()
        for bet in bets:
            w.writerow({
                "date":       date_str,
                "fixture_id": fixture_id,
                "match":      match_label,
                "issue":      bet["issue"],
                "cote":       bet["cote"],
                "ev_%":       bet["EV_%"],
                "kelly_%":    bet["mise_kelly_%"],
                "résultat":   "",
                "clv":        "",
            })
    logging.info(f"[Perf] {len(bets)} value bet(s) loggé(s) dans bet_history.csv")


# ─────────────────────────────────────────────
#  OUTILS
# ─────────────────────────────────────────────

def get_fixtures_today() -> str:
    today   = date.today().isoformat()
    all_fix = _api_get("fixtures", {"date": today})
    result: list[dict] = []
    for m in all_fix:
        lid = m["league"]["id"]
        if lid not in TARGET_LEAGUES:
            continue
        ts = m["fixture"]["timestamp"]
        result.append({
            "fixture_id":       m["fixture"]["id"],
            "tournament":       TARGET_LEAGUES[lid],
            "league_id":        lid,
            "is_neutral_venue": lid in NEUTRAL_LEAGUES,
            "heure_fr":         _ts_to_fr(ts) if ts else "?",
            "home_team":        m["teams"]["home"]["name"],
            "home_id":          m["teams"]["home"]["id"],
            "away_team":        m["teams"]["away"]["name"],
            "away_id":          m["teams"]["away"]["id"],
            "statut":           m["fixture"]["status"]["long"],
        })
    if not result:
        return "Aucun match trouvé aujourd'hui pour les ligues sélectionnées."
    return json.dumps(result, ensure_ascii=False, indent=2)


def get_match_analysis(
    fixture_id: int,
    is_neutral_venue: bool,
    league_id: int,
    home_team: str = "",
    away_team: str = "",
) -> str:
    """
    Analyse complète orientée value betting :
    1. Forme (5 derniers matchs) via /predictions
    2. H2H — 3 matchs récents seulement (indicateur faible, pondéré en conséquence)
    3. Poisson forces relatives + shrinkage → λ_dom, λ_ext
    4. Probas 1X2 du modèle vs probas implicites déviggées
    5. Value bets (edge > 3%, EV > 4%) avec mise Kelly/4
    6. Avertissements QA si données insuffisantes
    """
    time.sleep(0.3)

    # ── 1. Prédictions API (forme + H2H) ──────────────────────────
    preds    = _api_get("predictions", {"fixture": fixture_id})
    forme    = {}
    h2h_list = []
    lam_h = lam_a = 0.0
    poisson_data: dict = {}
    p_dom = p_nul = p_ext = None
    h_played = a_played = 0

    if preds:
        p       = preds[0]
        teams   = p.get("teams", {})
        h_last5 = teams.get("home", {}).get("last_5", {})
        a_last5 = teams.get("away", {}).get("last_5", {})

        h_avg_for = float(h_last5.get("goals", {}).get("for",     {}).get("average", 0) or 0)
        h_avg_ag  = float(h_last5.get("goals", {}).get("against", {}).get("average", 0) or 0)
        a_avg_for = float(a_last5.get("goals", {}).get("for",     {}).get("average", 0) or 0)
        a_avg_ag  = float(a_last5.get("goals", {}).get("against", {}).get("average", 0) or 0)
        h_played  = int(h_last5.get("played") or 0)
        a_played  = int(a_last5.get("played") or 0)

        forme = {
            "domicile": {
                "matchs_joués":       h_played,
                "buts_marqués_moy":   h_avg_for,
                "buts_encaissés_moy": h_avg_ag,
            },
            "extérieur": {
                "matchs_joués":       a_played,
                "buts_marqués_moy":   a_avg_for,
                "buts_encaissés_moy": a_avg_ag,
            },
        }

        # H2H : 3 derniers matchs seulement
        # Note : indicateur à faible valeur prédictive (effectifs changent, matchs anciens)
        # Ne pas utiliser comme argument principal du pick
        h2h_list = [
            {
                "date":  m["fixture"]["date"][:10],
                "match": (
                    f"{m['teams']['home']['name']} "
                    f"{m['goals']['home']}-{m['goals']['away']} "
                    f"{m['teams']['away']['name']}"
                ),
            }
            for m in p.get("h2h", [])[:3]
        ]

        # Poisson avec forces relatives + shrinkage
        lam_h, lam_a = _relative_poisson(
            h_avg_for, h_avg_ag, a_avg_for, a_avg_ag,
            h_played, a_played, league_id, is_neutral_venue,
        )
        p_dom, p_nul, p_ext = _poisson_1x2(lam_h, lam_a)

        poisson_data = {
            "λ_dom":               lam_h,
            "λ_ext":               lam_a,
            "buts_attendus_total": round(lam_h + lam_a, 2),
            "p_victoire_dom_%":    round(p_dom * 100, 1),
            "p_nul_%":             round(p_nul * 100, 1),
            "p_victoire_ext_%":    round(p_ext * 100, 1),
            "top_scores":          _top_scores(lam_h, lam_a),
        }

    # ── 2. Cotes bookmaker (1X2) ──────────────────────────────────
    cote_dom = cote_nul = cote_ext = None
    try:
        odds_list = _api_get("odds", {"fixture": fixture_id})
        if odds_list:
            for bk in odds_list[0].get("bookmakers", [])[:1]:
                for bet in bk.get("bets", []):
                    if bet["name"] == "Match Winner":
                        for v in bet.get("values", []):
                            if   v["value"] == "Home": cote_dom = float(v["odd"])
                            elif v["value"] == "Draw": cote_nul = float(v["odd"])
                            elif v["value"] == "Away": cote_ext = float(v["odd"])
    except Exception as e:
        logging.warning(f"Erreur cotes fixture {fixture_id}: {e}")

    # ── 3. Value betting ──────────────────────────────────────────
    value_bets   = []
    marche_devig = {}
    if all(x is not None for x in [cote_dom, cote_nul, cote_ext, p_dom]):
        fair = _devig(cote_dom, cote_nul, cote_ext)
        marche_devig = {
            "p_dom_%":  round(fair["p_dom"] * 100, 1),
            "p_nul_%":  round(fair["p_nul"] * 100, 1),
            "p_ext_%":  round(fair["p_ext"] * 100, 1),
            "marge_%":  fair["marge_%"],
        }
        value_bets = _compute_value_bets(p_dom, p_nul, p_ext, cote_dom, cote_nul, cote_ext)

    # ── 4. QA ─────────────────────────────────────────────────────
    warnings: list[str] = []
    if h_played < 3 or a_played < 3:
        warnings.append(
            f"Petit échantillon ({h_played}/{a_played} matchs) — estimation Poisson fragile, "
            "shrinkage augmenté vers la moyenne de ligue."
        )
    if cote_dom is None:
        warnings.append("Cotes non disponibles — calcul EV impossible, aucun value bet.")

    # Log automatique pour suivi performances
    match_label = f"{home_team} vs {away_team}" if home_team else str(fixture_id)
    _log_bets_to_csv(date.today().isoformat(), match_label, fixture_id, value_bets)

    return json.dumps({
        "forme":          forme,
        "h2h":            h2h_list,
        "poisson":        poisson_data,
        "cotes_brutes":   {"dom": cote_dom, "nul": cote_nul, "ext": cote_ext},
        "marché_dévig":   marche_devig,
        "value_bets":     value_bets,
        "avertissements": warnings,
    }, ensure_ascii=False, indent=2)


def send_telegram_report(text: str) -> str:
    url    = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    chunks = [text[i:i + 4000] for i in range(0, len(text), 4000)]
    for chunk in chunks:
        resp = requests.post(
            url,
            json={"chat_id": TELEGRAM_CHAT_ID, "text": chunk, "parse_mode": "HTML"},
            timeout=15,
        )
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
            "Retourne fixture_id, league_id, home/away team + IDs, heure FR, is_neutral_venue. "
            "Toujours appeler EN PREMIER, sans argument."
        ),
        "input_schema": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "get_match_analysis",
        "description": (
            "Analyse complète orientée value betting. "
            "Retourne : forme, H2H (3 matchs récents, faible poids prédictif), "
            "probabilités Poisson du modèle, marché dévig, et value_bets. "
            "value_bets = liste des paris où modèle > marché (edge > 3%, EV > 4%). "
            "Si value_bets est vide → aucune valeur sur ce match, ne pas forcer de pick. "
            "Appeler avec home_team et away_team pour le suivi de performance."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "fixture_id": {
                    "type": "integer",
                    "description": "ID du match (fourni par get_fixtures_today)",
                },
                "is_neutral_venue": {
                    "type": "boolean",
                    "description": "true pour la Coupe du Monde (terrain neutre)",
                },
                "league_id": {
                    "type": "integer",
                    "description": "ID de la ligue (fourni par get_fixtures_today)",
                },
                "home_team": {
                    "type": "string",
                    "description": "Nom de l'équipe à domicile (pour le suivi perf)",
                },
                "away_team": {
                    "type": "string",
                    "description": "Nom de l'équipe à l'extérieur (pour le suivi perf)",
                },
            },
            "required": ["fixture_id", "is_neutral_venue", "league_id"],
        },
    },
    {
        "name": "send_telegram_report",
        "description": (
            "Envoie le rapport final sur Telegram (HTML : <b>gras</b>, <i>italique</i>). "
            "Appeler UNE SEULE FOIS à la fin avec l'intégralité du rapport."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "text": {
                    "type": "string",
                    "description": "Rapport complet formaté en HTML",
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
            case "get_match_analysis":
                return get_match_analysis(
                    fixture_id       = tool_input["fixture_id"],
                    is_neutral_venue = tool_input.get("is_neutral_venue", False),
                    league_id        = tool_input.get("league_id", 39),
                    home_team        = tool_input.get("home_team", ""),
                    away_team        = tool_input.get("away_team", ""),
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
    today = date.today().strftime("%d/%m/%Y")
    return f"""Tu es un analyste sportif qui rédige des rapports de pronostics accessibles à tous.
Date : {today}.

Le moteur de décision probabiliste tourne en Python. Ton rôle : orchestrer les appels d'outils,
puis rédiger un rapport que n'importe qui peut comprendre — même sans connaître les statistiques.

━━━━━━━━━━━━━━━━━━━━━━━━
ÉTAPES OBLIGATOIRES
━━━━━━━━━━━━━━━━━━━━━━━━
1. Appelle get_fixtures_today (sans argument).
2. Si aucun match → envoie un Telegram le signalant et arrête.
3. Sélection :
   - FIFA Coupe du Monde : TOUS les matchs.
   - Autres ligues : max 3 matchs par ligue, 6 au total.
4. Pour chaque match : appelle get_match_analysis avec fixture_id, is_neutral_venue, league_id,
   home_team et away_team (fournis par get_fixtures_today).
5. Génère le rapport et appelle send_telegram_report UNE SEULE FOIS.

━━━━━━━━━━━━━━━━━━━━━━━━
RÈGLE CENTRALE — VALUE BETTING
━━━━━━━━━━━━━━━━━━━━━━━━
- Chaque analyse retourne value_bets : paris où notre modèle détecte un avantage réel sur le marché.
- Si value_bets est vide → affiche "Aucun pari recommandé" — N'INVENTE PAS de pick.
- N'utilise PAS le H2H comme argument principal (effectifs changent, trop peu de matchs).
- La mise recommandée est pré-calculée (Kelly/4) — utilise-la telle quelle.
- Tu expliques les données, tu ne prends pas de décision : le modèle l'a déjà fait.

━━━━━━━━━━━━━━━━━━━━━━━━
FORMAT DU RAPPORT (HTML strict, Telegram)
━━━━━━━━━━━━━━━━━━━━━━━━

<b>⚽ PRONOSTICS DU {today}</b>
<i>Analyse statistique indépendante des bookmakers</i>

━━━━━━━━━━━━━━━━━━━━━━━━

[Si matchs WC présents]
<b>🌍 FIFA COUPE DU MONDE 2026</b>

[Si matchs de ligues européennes]
<b>🏆 LIGUES EUROPÉENNES</b>

─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─

Pour CHAQUE match, reproduis exactement cette structure :

<b>🏟 [Éq. A] vs [Éq. B]</b>
<i>[Compétition] — 🕐 [Heure]</i>

📊 <b>Forme récente (5 derniers matchs) :</b>
  • [Éq. A] : [x] buts marqués en moyenne, [x] encaissés par match
  • [Éq. B] : [x] buts marqués en moyenne, [x] encaissés par match

🔢 <b>Probabilités selon notre modèle :</b>
  • Victoire [Éq. A] : <b>[X]%</b>   ← bookmakers donnent [X]%
  • Match nul : <b>[X]%</b>           ← bookmakers donnent [X]%
  • Victoire [Éq. B] : <b>[X]%</b>   ← bookmakers donnent [X]%
  <i>Score le plus probable : [X-X] ([Y]% de chances)</i>

📝 <b>Analyse :</b>
[OBLIGATOIRE — 3 à 5 phrases en langage naturel. Tu dois expliquer :
 1. Quelle équipe est en meilleure forme et pourquoi, en citant les chiffres de forme (buts marqués/encaissés).
 2. Ce que voit notre modèle : y a-t-il un écart entre nos probabilités et celles des bookmakers ?
    Si oui, sur quelle issue, et pourquoi c'est potentiellement intéressant.
 3. Conclusion claire : le match vaut-il la peine d'être joué, et sur quelle issue ?
 Ton ton doit être celui d'un ami qui explique le match simplement, sans jargon technique.
 Exemples de formulations à utiliser :
 - "L'Allemagne est en grande forme offensive (3,3 buts par match) face à un Paraguay qui peine à scorer."
 - "Notre modèle donne 31% de chances au Japon, alors que les bookmakers ne lui en accordent que 19% — c'est là qu'on voit de la valeur."
 - "Les deux équipes se neutralisent offensivement, ce qui rend le résultat très incertain."
 - "Aucun écart significatif entre notre modèle et les cotes : pas d'avantage à parier ici."]

[Si value_bets non vide :]
💰 <b>Pari recommandé :</b>
[Pour chaque value bet :]
  ✅ <b>[Victoire [Éq. X] / Match nul]</b> @ cote [X.XX]
  Notre modèle : [X]% · Bookmakers (réel) : [X]% · Avantage : +[X] points
  Mise conseillée : [X]% de ta bankroll habituelle

[Si value_bets vide :]
🚫 <b>Aucun pari recommandé sur ce match</b>
<i>Les probabilités de notre modèle sont trop proches des cotes du marché pour justifier un risque.</i>

[Si avertissements présents :]
<i>ℹ️ [Reformule l'avertissement en clair. Ex: "Attention : ces équipes ont peu de matchs joués, l'estimation est moins précise qu'à l'habitude."]</i>

━━━━━━━━━━━━━━━━━━━━━━━━

<b>📋 RÉSUMÉ DES PARIS DU JOUR</b>
[Tous les value bets de la journée, triés par avantage décroissant.
 Si aucun → "Aucun pari recommandé aujourd'hui — le marché est bien calibré sur l'ensemble des matchs analysés."]

1. [Match] — [Victoire X / Nul] @ [cote] · Notre modèle : [X]% vs Bookmakers : [X]% · Mise : [X]%
…

<i>⚠️ Ces analyses sont à titre informatif. Ne pariez que ce que vous pouvez vous permettre de perdre.</i>

━━━━━━━━━━━━━━━━━━━━━━━━
CONSIGNES DE RÉDACTION
━━━━━━━━━━━━━━━━━━━━━━━━
- heure_fr est déjà en heure française (UTC+2) — inutile de la convertir
- Terrain neutre (WC) : utilise les noms des équipes, jamais "domicile"/"extérieur"
- Les % "bookmakers" dans la section probabilités = valeurs de marché_dévig (marge déjà retirée)
- Traduis les issues : "1 (domicile)" → "Victoire [nom équipe]", "X (nul)" → "Match nul", "2 (extérieur)" → "Victoire [nom équipe]"
- Évite tout jargon : pas de "lambda", "shrinkage", "EV", "edge", "dévigging" dans le rapport final
- Le paragraphe Analyse est OBLIGATOIRE pour chaque match — c'est la valeur ajoutée principale
- Sois direct et factuel, évite les formulations vagues ("match équilibré", "tout est possible")
"""


# ─────────────────────────────────────────────
#  BOUCLE DE L'AGENT
# ─────────────────────────────────────────────

def run_agent(max_steps: int = 60) -> None:
    today_str     = date.today().strftime("%d/%m/%Y")
    system_prompt = _build_system_prompt()

    messages = [{
        "role": "user",
        "content": f"Analyse les matchs de football du {today_str} et envoie le rapport complet sur Telegram.",
    }]

    logging.info("=" * 55)
    logging.info(f"AGENT FOOT — VALUE BETTING — {today_str}")
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

    logging.warning("⚠️ Nombre maximum d'étapes atteint.")


# ─────────────────────────────────────────────
#  LANCEMENT
# ─────────────────────────────────────────────

if __name__ == "__main__":
    run_agent()

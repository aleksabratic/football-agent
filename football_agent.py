"""
============================================================
 AGENT IA — PRÉDICTIONS FOOTBALLISTIQUES + TELEGRAM
 (API-Football api-sports.io — plan Free 100 req/jour)
============================================================

Cet agent :
  1. Récupère les matchs du jour via API-Football
  2. Analyse chaque match : forme, H2H, stats Poisson, cotes
  3. Génère des prédictions via Claude et envoie sur Telegram

Pré-requis :
    pip install anthropic requests python-dotenv

Variables d'environnement (.env) :
    ANTHROPIC_API_KEY   → console.anthropic.com
    APIFOOTBALL_KEY     → api-sports.io (plan Free = 100 req/jour)
    TELEGRAM_BOT_TOKEN  → @BotFather sur Telegram
    TELEGRAM_CHAT_ID    → ton ID de chat

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

ANTHROPIC_API_KEY   = os.environ["ANTHROPIC_API_KEY"]
APIFOOTBALL_KEY     = os.environ["APIFOOTBALL_KEY"]
TELEGRAM_BOT_TOKEN  = os.environ["TELEGRAM_BOT_TOKEN"]
TELEGRAM_CHAT_ID    = os.environ["TELEGRAM_CHAT_ID"]

APIFOOTBALL_BASE    = "https://v3.football.api-sports.io"
APIFOOTBALL_HEADERS = {"x-apisports-key": APIFOOTBALL_KEY}

# IDs de ligues API-Football → nom affiché
TARGET_LEAGUES: dict[int, str] = {
    1:   "FIFA Coupe du Monde 2026",
    39:  "Premier League",
    140: "La Liga",
    78:  "Bundesliga",
    135: "Serie A",
    61:  "Ligue 1",
}

# Ligues jouées sur terrain neutre
NEUTRAL_LEAGUES: set[int] = {1}

_TZ_FR = timezone(timedelta(hours=2))

MODEL  = "claude-opus-4-8"
client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)


# ─────────────────────────────────────────────
#  HELPERS
# ─────────────────────────────────────────────

def _api_get(endpoint: str, params: dict = None) -> list:
    """Appel GET vers API-Football avec retry sur 429."""
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


def _poisson_prob(lam: float, k: int) -> float:
    if lam <= 0:
        return 1.0 if k == 0 else 0.0
    return math.exp(-lam) * (lam ** k) / math.factorial(k)


# ─────────────────────────────────────────────
#  OUTILS
# ─────────────────────────────────────────────

def get_fixtures_today() -> str:
    """
    Récupère tous les matchs du jour pour les ligues cibles.
    Retourne fixture_id, IDs équipes, heure, ligue, terrain neutre.
    """
    today    = date.today().isoformat()
    all_fix  = _api_get("fixtures", {"date": today})

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


def get_match_analysis(fixture_id: int, is_neutral_venue: bool) -> str:
    """
    Récupère en un seul appel toutes les données d'analyse d'un match :
    - Forme des 5 derniers matchs (home et away)
    - H2H complet
    - Stats comparatives (forme %, attaque, défense, Poisson)
    - Prédiction du vainqueur et conseil
    - Cotes 1X2 (bookmaker)
    Puis calcule le score exact le plus probable via Poisson local.
    """
    time.sleep(0.3)

    # 1. Prédictions (forme + H2H + comparaison + conseil)
    preds = _api_get("predictions", {"fixture": fixture_id})
    pred_data: dict = {}
    if preds:
        p = preds[0]
        teams = p.get("teams", {})
        h_last5 = teams.get("home", {}).get("last_5", {})
        a_last5 = teams.get("away", {}).get("last_5", {})
        comp    = p.get("comparison", {})

        # Calcul Poisson depuis les moyennes de buts
        h_avg_for  = float(h_last5.get("goals", {}).get("for",     {}).get("average", 0) or 0)
        h_avg_ag   = float(h_last5.get("goals", {}).get("against", {}).get("average", 0) or 0)
        a_avg_for  = float(a_last5.get("goals", {}).get("for",     {}).get("average", 0) or 0)
        a_avg_ag   = float(a_last5.get("goals", {}).get("against", {}).get("average", 0) or 0)

        lam_home = (h_avg_for + a_avg_ag) / 2
        lam_away = (a_avg_for + h_avg_ag) / 2
        if not is_neutral_venue:
            lam_home *= 1.15

        scores = []
        for g1 in range(6):
            for g2 in range(6):
                prob = _poisson_prob(lam_home, g1) * _poisson_prob(lam_away, g2) * 100
                scores.append((g1, g2, round(prob, 2)))
        scores.sort(key=lambda x: x[2], reverse=True)
        top_scores = [{"score": f"{g1}-{g2}", "probabilité_%": p_} for g1, g2, p_ in scores[:5]]

        pred_data = {
            "forme_domicile_5j": {
                "matchs_joués":    h_last5.get("played"),
                "forme_%":         h_last5.get("form"),
                "buts_marqués_moy": h_avg_for,
                "buts_encaissés_moy": h_avg_ag,
            },
            "forme_extérieur_5j": {
                "matchs_joués":    a_last5.get("played"),
                "forme_%":         a_last5.get("form"),
                "buts_marqués_moy": a_avg_for,
                "buts_encaissés_moy": a_avg_ag,
            },
            "h2h_matchs": [
                {
                    "date":   m["fixture"]["date"][:10],
                    "match":  f"{m['teams']['home']['name']} {m['goals']['home']}-{m['goals']['away']} {m['teams']['away']['name']}",
                }
                for m in p.get("h2h", [])[:5]
            ],
            "comparaison": {
                "forme_%":              comp.get("form"),
                "attaque_%":            comp.get("att"),
                "défense_%":            comp.get("def"),
                "poisson_api_%":        comp.get("poisson_distribution"),
                "h2h_%":                comp.get("h2h"),
                "total_score_%":        comp.get("total"),
            },
            "prédiction_api": {
                "vainqueur":    p.get("predictions", {}).get("winner"),
                "conseil":      p.get("predictions", {}).get("advice"),
                "percent":      p.get("predictions", {}).get("percent"),
            },
            "poisson_local": {
                "buts_attendus_dom": round(lam_home, 2),
                "buts_attendus_ext": round(lam_away, 2),
                "terrain_neutre":    is_neutral_venue,
                "top_scores":        top_scores,
            },
        }

    # 2. Cotes
    odds_result: dict = {}
    try:
        odds_list = _api_get("odds", {"fixture": fixture_id})
        if odds_list:
            for bk in odds_list[0].get("bookmakers", [])[:1]:
                for bet in bk.get("bets", []):
                    if bet["name"] == "Match Winner":
                        for v in bet.get("values", []):
                            odds_result[f"cote_{v['value']}"] = float(v["odd"])
                    elif "Goals Over/Under" in bet["name"]:
                        for v in bet.get("values", []):
                            if "2.5" in str(v.get("value", "")):
                                label = f"cote_{v['value'].lower().replace(' ', '_')}_2.5"
                                odds_result[label] = float(v["odd"])
    except Exception as e:
        odds_result["erreur_cotes"] = str(e)

    return json.dumps({
        "analyse": pred_data,
        "cotes":   odds_result if odds_result else "non disponibles",
    }, ensure_ascii=False, indent=2)


def send_telegram_report(text: str) -> str:
    """Envoie le rapport HTML sur Telegram (chunks de 4000 chars)."""
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
            "Retourne fixture_id, home/away team + IDs, heure française, is_neutral_venue. "
            "Toujours appeler EN PREMIER, sans argument."
        ),
        "input_schema": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "get_match_analysis",
        "description": (
            "Récupère toutes les données d'analyse d'un match en un seul appel : "
            "forme des 5 derniers matchs, H2H, stats comparatives (attaque/défense/Poisson), "
            "prédiction de l'API, score exact Poisson calculé localement, et cotes 1X2. "
            "Appeler pour chaque match après get_fixtures_today."
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
                    "description": "true pour la Coupe du Monde (terrain neutre), false pour les ligues",
                },
            },
            "required": ["fixture_id", "is_neutral_venue"],
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
    return f"""Tu es un expert en analyse footballistique. Ta mission du {today} :

ÉTAPES OBLIGATOIRES :
1. Appelle get_fixtures_today (sans argument).
2. S'il n'y a aucun match → envoie un message Telegram le signalant et arrête-toi.
3. Sélection des matchs :
   - FIFA Coupe du Monde : TOUS les matchs du jour sans exception.
   - Autres ligues : max 3 matchs par ligue, 6 au total.
4. Pour chaque match : appelle get_match_analysis(fixture_id, home_id, away_id, is_neutral_venue).
5. Génère le rapport et appelle send_telegram_report UNE SEULE FOIS.

RÈGLE COUPE DU MONDE :
- Terrain neutre : ne pas mentionner d'avantage domicile/extérieur.
- Utilise les noms des équipes directement (pas "domicile" / "extérieur").

FORMAT DU RAPPORT TELEGRAM (HTML strict) :

<b>⚽ PRONOSTICS DU {today}</b>
<i>Analyse : forme · H2H · Poisson · cotes</i>

━━━━━━━━━━━━━━━━━━━━━━━━

Si matchs WC présents :
<b>🌍 FIFA COUPE DU MONDE 2026</b>

Puis matchs de clubs :
<b>🏆 LIGUES EUROPÉENNES</b>

Pour chaque match :
<b>🏟 [Équipe A] vs [Équipe B]</b>
<i>[Compétition] — 🕐 [Heure]</i>

📊 <b>Forme récente :</b>
  [Éq. A] : forme [X]% — [X] buts/match marqués / [X] encaissés
  [Éq. B] : forme [X]% — [X] buts/match marqués / [X] encaissés

⚖️ <b>H2H :</b> [résumé des dernières confrontations]

📈 <b>Stats comparatives :</b> attaque [X]%/[X]% · défense [X]%/[X]% · Poisson [X]%/[X]%

💹 <b>Cotes :</b> [Éq. A] [X.XX] | Nul [X.XX] | [Éq. B] [X.XX]

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

RÈGLES :
- heure_fr est déjà en UTC+2 (heure française)
- Sois factuel et concis, utilise les noms réels des équipes
- Le score Poisson est une estimation statistique
"""


# ─────────────────────────────────────────────
#  BOUCLE DE L'AGENT
# ─────────────────────────────────────────────

def run_agent(max_steps: int = 60) -> None:
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

    logging.warning("⚠️ Nombre maximum d'étapes atteint.")


# ─────────────────────────────────────────────
#  LANCEMENT
# ─────────────────────────────────────────────

if __name__ == "__main__":
    run_agent()

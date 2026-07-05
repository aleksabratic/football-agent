"""
============================================================
 AGENT IA — PRONOSTICS FOOT (VALUE BETTING)
 (API-Football api-sports.io — plan Free 100 req/jour, 10 req/min)
============================================================

Architecture de décision :
  - Moteur probabiliste en Python :
      * Forces d'attaque/défense sur les stats SAISON avec splits
        domicile/extérieur (fournies par /predictions, zéro requête en plus)
      * Ajustement de forme récente (5 derniers matchs, ±15 % max)
      * Shrinkage vers le prior de ligue quand l'échantillon est petit
      * Correction Dixon-Coles (ρ) : corrige la sous-estimation des nuls
        du Poisson indépendant
      * Matchs internationaux : λ ancrés à 70 % sur le classement Elo
        mondial (eloratings.net), bonus pays hôte WC 2026 (USA/MEX/CAN)
  - Marché :
      * Bookmaker de référence (Pinnacle si dispo) dévigué par la
        méthode "power" (moins biaisée que la multiplicative)
      * Meilleure cote disponible parmi tous les books pour l'EV
  - Probabilité finale = blend 65 % marché / 35 % modèle
    (le marché sert de prior : on ne parie que si le désaccord résiduel
     survit au blend)
  - Marchés couverts : 1X2, Over/Under 2.5, BTTS
  - Pick uniquement si edge ≥ 2 % ET EV ≥ 3 % (seuils doublés en Coupe
    du Monde : stats internationales peu fiables)
  - Mise = Kelly/4, plafonnée à 2 % par pari et 8 % par jour
  - Suivi : bet_history.csv (commité par le workflow GitHub), settlement
    automatique des paris en attente à chaque run + bilan (ROI, yield,
    Brier) injecté dans le rapport
  - Mon Petit Prono : score conseillé maximisant les points attendus
    (élimination directe : distribution des scores étendue à 120 min,
     règlement sur le score final hors tirs au but)
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
import re
import json
import math
import time
import logging
import unicodedata
import requests
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
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

# ── Modèle ────────────────────────────────────
RHO           = -0.10   # correction Dixon-Coles (gonfle 0-0 / 1-1, dégonfle 1-0 / 0-1)
MARKET_WEIGHT = 0.65    # poids du marché dans la proba finale (le modèle pèse 35 %)
FORM_CLAMP    = 0.15    # ajustement max de la forme récente (±15 %)

# ── Value betting ─────────────────────────────
EDGE_THRESHOLD    = 0.02   # avantage minimum APRÈS blend marché/modèle (2 %)
EV_THRESHOLD      = 0.03   # espérance de gain minimum (3 %)
WC_THRESHOLD_MULT = 2.0    # seuils doublés en Coupe du Monde (stats peu fiables)
KELLY_FRACTION    = 0.25   # quart de Kelly pour limiter la variance
MAX_STAKE_PER_BET = 0.02   # 2 % de bankroll max par pari
DAILY_EXPOSURE_CAP = 0.08  # 8 % de bankroll max engagés par jour

# ── Sélection des matchs (budget 100 req/jour, 2 req/match) ──
MAX_MATCHES_PER_DAY   = 15
MAX_MATCHES_PER_LEAGUE = 3  # hors Coupe du Monde (WC = tous les matchs)

# ── Mon Petit Prono (barème par défaut : ajuster selon ta ligue) ──
MPP_PTS_RESULT = 1   # points pour le bon résultat (1/N/2)
MPP_PTS_EXACT  = 3   # points pour le score exact (inclut le bon résultat)

# ── Elo des sélections nationales (eloratings.net) ──────────────
# Pour les matchs internationaux, 4-5 matchs de stats API sont du bruit :
# l'Elo (historique complet pondéré par la force des adversaires) est
# beaucoup plus fiable. Les λ finaux = blend Elo / stats.
ELO_RATINGS_URL  = "https://www.eloratings.net/World.tsv"
ELO_NAMES_URL    = "https://www.eloratings.net/en.teams.tsv"
ELO_HTTP_HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
ELO_WEIGHT       = 0.70  # poids de l'Elo dans les buts attendus (internationaux)
ELO_HOST_BONUS   = 80    # WC 2026 : USA/Mexique/Canada jouent réellement chez eux
ELO_HOST_NATIONS = {"usa", "united states", "mexico", "canada"}
# Noms API-Football → noms eloratings (formes normalisées) pour les cas ambigus
ELO_ALIASES = {
    "usa":                  "united states",
    "ivory coast":          "cote divoire",
    "czech republic":       "czechia",
    "cape verde islands":   "cape verde",
    "turkiye":              "turkey",
    "korea republic":       "south korea",
    "ir iran":              "iran",
}

# ── Bookmakers de référence pour le devig (ordre de préférence) ──
# Pinnacle = marché le plus efficient ; sinon Bet365 / Marathonbet / 1xBet
PREFERRED_BOOKMAKER_IDS = [4, 8, 2, 11]

# Marchés suivis : nom API → (label interne, mapping sélection API → sélection interne)
MARKETS_MAP = {
    "Match Winner":     ("1X2",     {"Home": "1", "Draw": "X", "Away": "2"}),
    "Goals Over/Under": ("O/U 2.5", {"Over 2.5": "Over", "Under 2.5": "Under"}),
    "Both Teams Score": ("BTTS",    {"Yes": "Oui", "No": "Non"}),
}
MARKET_SELECTIONS = {
    "1X2":     ["1", "X", "2"],
    "O/U 2.5": ["Over", "Under"],
    "BTTS":    ["Oui", "Non"],
}

PERF_FILE = Path(__file__).parent / "bet_history.csv"
PERF_FIELDS = [
    "date", "fixture_id", "match", "marché", "pari", "cote", "bookmaker",
    "p_modèle_%", "p_marché_%", "p_finale_%", "edge_%", "ev_%", "mise_%",
    "résultat", "gain_unités", "clv",
]

# Historique des pronos Mon Petit Prono (réglés comme les paris)
MPP_FILE = Path(__file__).parent / "mpp_history.csv"
MPP_FIELDS = [
    "date", "fixture_id", "match", "score_conseillé", "points_attendus",
    "score_réel", "points", "résultat",
]

_TZ_FR = ZoneInfo("Europe/Paris")
MODEL  = "claude-opus-4-8"
client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

# Exposition cumulée sur la journée (réinitialisée à chaque run)
_RUN_EXPOSURE = 0.0


# ─────────────────────────────────────────────
#  HELPER API
# ─────────────────────────────────────────────

_last_api_call = 0.0

def _throttle() -> None:
    """Plan Free = 10 req/min → au moins 6,5 s entre deux appels."""
    global _last_api_call
    wait = 6.5 - (time.monotonic() - _last_api_call)
    if wait > 0:
        time.sleep(wait)
    _last_api_call = time.monotonic()


def _api_get(endpoint: str, params: dict = None) -> list:
    url   = f"{APIFOOTBALL_BASE}/{endpoint}"
    delay = 10
    for attempt in range(4):
        _throttle()
        resp = requests.get(url, headers=APIFOOTBALL_HEADERS, params=params or {}, timeout=15)
        if resp.status_code == 429:
            if attempt < 3:
                logging.warning(f"429 rate limit — attente {delay}s (retry {attempt+2}/4)...")
                time.sleep(delay)
                delay *= 2
                continue
            # Ne JAMAIS retourner [] en silence : le rapport dirait
            # "aucun match" alors que l'API est en panne.
            raise RuntimeError(f"API-Football : rate limit persistant sur /{endpoint}")
        resp.raise_for_status()
        data = resp.json()
        if data.get("errors"):
            raise RuntimeError(f"API errors sur /{endpoint}: {data['errors']}")
        return data.get("response", [])
    raise RuntimeError(f"API-Football : échec répété sur /{endpoint}")


def _f(x, default: float = 0.0) -> float:
    """Conversion float tolérante (l'API renvoie des strings, parfois null)."""
    try:
        return float(x)
    except (TypeError, ValueError):
        return default


def _now_fr() -> datetime:
    return datetime.now(tz=_TZ_FR)


# ─────────────────────────────────────────────
#  MOTEUR POISSON + DIXON-COLES
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
    Évite qu'un petit échantillon hallucine un grand favori.
    """
    w = min(n_games, 10) / 10.0
    return w * observed + (1.0 - w) * prior


def _form_factor(l5_played: int, l5_gf_avg: float, season_gf_avg: float) -> float:
    """
    Ajustement de forme : ratio buts/match sur les 5 derniers vs la saison,
    clampé à ±15 % et pondéré par le nombre de matchs récents disponibles.
    """
    if l5_played < 3 or season_gf_avg <= 0.2:
        return 1.0
    ratio = max(1.0 - FORM_CLAMP, min(1.0 + FORM_CLAMP, l5_gf_avg / season_gf_avg))
    w = min(l5_played, 5) / 5.0
    return 1.0 + w * (ratio - 1.0)


def _team_stats(block: dict) -> dict:
    """Extrait les stats saison (splits dom/ext) + forme récente de /predictions."""
    last5 = block.get("last_5") or {}
    lg    = block.get("league") or {}
    fx    = (lg.get("fixtures") or {}).get("played") or {}
    gf    = ((lg.get("goals") or {}).get("for") or {}).get("average") or {}
    ga    = ((lg.get("goals") or {}).get("against") or {}).get("average") or {}
    l5g   = last5.get("goals") or {}
    return {
        "n_home":  int(fx.get("home") or 0),
        "n_away":  int(fx.get("away") or 0),
        "n_total": int(fx.get("total") or 0),
        "gf_home": _f(gf.get("home")), "gf_away": _f(gf.get("away")), "gf_total": _f(gf.get("total")),
        "ga_home": _f(ga.get("home")), "ga_away": _f(ga.get("away")), "ga_total": _f(ga.get("total")),
        "l5_played": int(last5.get("played") or 0),
        "l5_gf":     _f((l5g.get("for") or {}).get("average")),
        "l5_ga":     _f((l5g.get("against") or {}).get("average")),
        "l5_forme":  last5.get("form"),
    }


def _expected_goals(home: dict, away: dict, league_id: int, is_neutral: bool) -> tuple[float, float, dict]:
    """
    Buts attendus (λ) via forces relatives sur les stats SAISON.

    Match classique : split domicile/extérieur — l'avantage du terrain est
    porté UNE SEULE fois, par les moyennes de ligue avg_h/avg_a et les splits.
      att_dom = buts marqués à domicile / moyenne dom de la ligue
      def_dom = buts encaissés à domicile / moyenne ext de la ligue
      λ_dom   = att_dom × def_ext × avg_h   (pas de prime ×1.10 en plus)

    Terrain neutre (WC) : stats totales, base commune = moyenne globale.
    Puis ajustement de forme récente (5 derniers matchs, ±15 %).
    """
    avg_h, avg_a = LEAGUE_GOALS.get(league_id, (1.40, 1.15))
    overall      = (avg_h + avg_a) / 2.0

    if is_neutral:
        att_h = _shrink(home["gf_total"] / overall, home["n_total"])
        def_h = _shrink(home["ga_total"] / overall, home["n_total"])
        att_a = _shrink(away["gf_total"] / overall, away["n_total"])
        def_a = _shrink(away["ga_total"] / overall, away["n_total"])
        lam_h = att_h * def_a * overall
        lam_a = att_a * def_h * overall
    else:
        att_h = _shrink(home["gf_home"] / avg_h, home["n_home"])
        def_h = _shrink(home["ga_home"] / avg_a, home["n_home"])
        att_a = _shrink(away["gf_away"] / avg_a, away["n_away"])
        def_a = _shrink(away["ga_away"] / avg_h, away["n_away"])
        lam_h = att_h * def_a * avg_h
        lam_a = att_a * def_h * avg_a

    f_h = _form_factor(home["l5_played"], home["l5_gf"], home["gf_total"])
    f_a = _form_factor(away["l5_played"], away["l5_gf"], away["gf_total"])
    lam_h = max(0.2, min(4.5, lam_h * f_h))
    lam_a = max(0.2, min(4.5, lam_a * f_a))

    details = {
        "force_attaque_dom": round(att_h, 2), "force_défense_dom": round(def_h, 2),
        "force_attaque_ext": round(att_a, 2), "force_défense_ext": round(def_a, 2),
        "facteur_forme_dom": round(f_h, 3),   "facteur_forme_ext": round(f_a, 3),
    }
    return round(lam_h, 3), round(lam_a, 3), details


def _score_matrix(lam_h: float, lam_a: float, size: int = 12) -> list[list[float]]:
    """
    Matrice des scores Poisson avec correction Dixon-Coles :
    le Poisson indépendant sous-estime les nuls (0-0, 1-1) — ρ négatif
    les regonfle et dégonfle 1-0/0-1, puis on renormalise à 1.
    """
    m = [[_poisson_prob(lam_h, i) * _poisson_prob(lam_a, j) for j in range(size)]
         for i in range(size)]
    m[0][0] *= 1.0 - lam_h * lam_a * RHO
    m[0][1] *= 1.0 + lam_h * RHO
    m[1][0] *= 1.0 + lam_a * RHO
    m[1][1] *= 1.0 - RHO
    total = sum(sum(row) for row in m)
    return [[v / total for v in row] for row in m]


def _matrix_markets(m: list[list[float]]) -> dict:
    """Dérive tous les marchés de la matrice des scores."""
    size = len(m)
    p1 = px = p2 = over25 = btts = 0.0
    for i in range(size):
        for j in range(size):
            p = m[i][j]
            if   i > j:  p1 += p
            elif i == j: px += p
            else:        p2 += p
            if i + j >= 3:          over25 += p
            if i >= 1 and j >= 1:   btts   += p
    top = sorted(
        ({"score": f"{i}-{j}", "probabilité_%": round(m[i][j] * 100, 2)}
         for i in range(7) for j in range(7)),
        key=lambda x: x["probabilité_%"], reverse=True,
    )[:5]
    return {"p1": p1, "px": px, "p2": p2, "over25": over25, "btts": btts, "top_scores": top}


# ─────────────────────────────────────────────
#  ELO SÉLECTIONS NATIONALES (eloratings.net)
# ─────────────────────────────────────────────

_ELO_CACHE: dict | None = None  # nom normalisé → (elo, rang mondial)


def _normalize_team(name: str) -> str:
    """Minuscules, sans accents ni ponctuation — pour matcher les deux sources."""
    s = unicodedata.normalize("NFD", name)
    s = "".join(c for c in s if unicodedata.category(c) != "Mn")
    s = re.sub(r"[^a-z0-9 ]", "", s.lower())
    return re.sub(r"\s+", " ", s).strip()


def _load_elo() -> dict:
    """
    Charge les classements Elo (1 seule fois par run, hors quota API-Football).
    World.tsv : rang \\t rang \\t code \\t elo \\t ...
    en.teams.tsv : code \\t nom principal \\t alias...
    En cas d'échec réseau → {} : le moteur retombe sur les stats seules.
    """
    global _ELO_CACHE
    if _ELO_CACHE is not None:
        return _ELO_CACHE
    try:
        r_names = requests.get(ELO_NAMES_URL, headers=ELO_HTTP_HEADERS, timeout=15)
        r_names.raise_for_status()
        r_names.encoding = "utf-8"
        code_names: dict[str, list[str]] = {}
        for line in r_names.text.splitlines():
            parts = line.split("\t")
            if len(parts) >= 2:
                code_names[parts[0]] = [p for p in parts[1:] if p]

        r_elo = requests.get(ELO_RATINGS_URL, headers=ELO_HTTP_HEADERS, timeout=15)
        r_elo.raise_for_status()
        r_elo.encoding = "utf-8"
        table: dict[str, tuple[int, int]] = {}
        for line in r_elo.text.splitlines():
            parts = line.split("\t")
            if len(parts) < 4:
                continue
            try:
                rank, code, elo = int(parts[0]), parts[2], int(parts[3])
            except ValueError:
                continue
            for name in code_names.get(code, []):
                table[_normalize_team(name)] = (elo, rank)
        _ELO_CACHE = table
        logging.info(f"[Elo] {len(table)} noms d'équipes chargés depuis eloratings.net")
    except Exception as e:
        logging.warning(f"[Elo] Chargement impossible : {e}")
        _ELO_CACHE = {}
    return _ELO_CACHE


def _elo_lookup(team_name: str) -> tuple[int, int] | None:
    """(elo, rang mondial) pour une équipe, ou None si introuvable."""
    table = _load_elo()
    if not table or not team_name:
        return None
    key = _normalize_team(team_name)
    # Lookup direct d'abord, alias en secours seulement
    return table.get(key) or table.get(ELO_ALIASES.get(key, ""))


def _elo_expected_lambdas(elo_h: int, elo_a: int, total_goals: float) -> tuple[float, float]:
    """
    Convertit un écart d'Elo en buts attendus, en cohérence avec notre propre
    matrice Dixon-Coles : on cherche par dichotomie l'écart de buts qui donne
    une espérance de victoire (P(V) + P(N)/2) égale à celle prédite par l'Elo.
    Le total de buts reste celui du prior de compétition.
    """
    target = 1.0 / (1.0 + 10 ** (-(elo_h - elo_a) / 400.0))
    lo, hi = -3.0, 3.0
    lam_h = lam_a = total_goals / 2.0
    for _ in range(30):
        mid   = (lo + hi) / 2.0
        lam_h = max(0.15, (total_goals + mid) / 2.0)
        lam_a = max(0.15, (total_goals - mid) / 2.0)
        mk    = _matrix_markets(_score_matrix(lam_h, lam_a))
        if mk["p1"] + mk["px"] / 2.0 < target:
            lo = mid
        else:
            hi = mid
    return round(lam_h, 3), round(lam_a, 3)


def _apply_elo(
    lam_h: float, lam_a: float,
    home_team: str, away_team: str,
    league_id: int,
) -> tuple[float, float, dict, list[str]]:
    """
    Matchs internationaux : blend λ = 70 % Elo + 30 % stats.
    Bonus hôte WC 2026 (USA/Mexique/Canada : vrais matchs à domicile).
    Si l'Elo est indisponible pour une équipe → stats seules + avertissement.
    """
    warnings: list[str] = []
    eh = _elo_lookup(home_team)
    ea = _elo_lookup(away_team)
    if not eh or not ea:
        missing = [t for t, e in [(home_team, eh), (away_team, ea)] if not e]
        warnings.append(
            f"Classement Elo introuvable pour {', '.join(missing)} — "
            "modèle basé sur les stats seules, fiabilité réduite."
        )
        return lam_h, lam_a, {}, warnings

    elo_h, rank_h = eh
    elo_a, rank_a = ea
    is_host = _normalize_team(home_team) in ELO_HOST_NATIONS
    elo_h_eff = elo_h + (ELO_HOST_BONUS if is_host else 0)

    avg_h, avg_a = LEAGUE_GOALS.get(league_id, (1.30, 1.20))
    lam_elo_h, lam_elo_a = _elo_expected_lambdas(elo_h_eff, elo_a, avg_h + avg_a)
    mk_elo = _matrix_markets(_score_matrix(lam_elo_h, lam_elo_a))

    blended_h = round(ELO_WEIGHT * lam_elo_h + (1.0 - ELO_WEIGHT) * lam_h, 3)
    blended_a = round(ELO_WEIGHT * lam_elo_a + (1.0 - ELO_WEIGHT) * lam_a, 3)

    elo_block = {
        "elo_dom":            elo_h,
        "rang_mondial_dom":   rank_h,
        "elo_ext":            elo_a,
        "rang_mondial_ext":   rank_a,
        "bonus_hôte_appliqué": is_host,
        "p_1x2_selon_elo_%": {
            "1": round(mk_elo["p1"] * 100, 1),
            "X": round(mk_elo["px"] * 100, 1),
            "2": round(mk_elo["p2"] * 100, 1),
        },
        "poids_elo_dans_le_modèle": ELO_WEIGHT,
    }
    return blended_h, blended_a, elo_block, warnings


# ─────────────────────────────────────────────
#  MON PETIT PRONO
# ─────────────────────────────────────────────

def _extend_matrix_120(m90: list[list[float]], lam_h: float, lam_a: float) -> tuple:
    """
    Distribution des scores après PROLONGATION (matchs à élimination directe).
    Les scores non nuls à 90' sont définitifs ; chaque score nul se prolonge
    avec une mini-matrice Poisson de 30 minutes (λ/3 par équipe).
    Retourne (matrice_120, q1, qx, q2) où q* = issue de la prolongation seule
    sachant qu'il y a prolongation (q1 = l'équipe à domicile passe devant, etc.).
    """
    size = len(m90)
    lam_eh, lam_ea = lam_h / 3.0, lam_a / 3.0
    et = [[_poisson_prob(lam_eh, a) * _poisson_prob(lam_ea, b) for b in range(size)]
          for a in range(size)]
    q1 = sum(et[a][b] for a in range(size) for b in range(size) if a > b)
    qx = sum(et[a][a] for a in range(size))
    q2 = max(0.0, 1.0 - q1 - qx)

    m120 = [[0.0] * size for _ in range(size)]
    for i in range(size):
        for j in range(size):
            p = m90[i][j]
            if p == 0.0:
                continue
            if i != j:
                m120[i][j] += p
            else:
                for a in range(size - i):
                    for b in range(size - j):
                        m120[i + a][j + b] += p * et[a][b]
    return m120, q1, qx, q2


def _mon_petit_prono(m: list[list[float]], p1: float, px: float, p2: float) -> dict:
    """
    Score à jouer sur Mon Petit Prono : maximise les points attendus.
    E(score) = P(bon résultat) × PTS_RESULT + P(score exact) × (PTS_EXACT − PTS_RESULT)
    (le score exact rapporte PTS_EXACT au total, résultat inclus).
    Le résultat le plus probable ne donne pas toujours le meilleur score :
    on balaye toute la grille 0-5.
    """
    cands = []
    for i in range(6):
        for j in range(6):
            p_res = p1 if i > j else (px if i == j else p2)
            e = p_res * MPP_PTS_RESULT + m[i][j] * (MPP_PTS_EXACT - MPP_PTS_RESULT)
            cands.append((e, i, j))
    cands.sort(key=lambda x: x[0], reverse=True)
    best = cands[0]
    return {
        "score_conseillé":  f"{best[1]}-{best[2]}",
        "points_attendus":  round(best[0], 2),
        "alternatives": [
            {"score": f"{e[1]}-{e[2]}", "points_attendus": round(e[0], 2)}
            for e in cands[1:3]
        ],
    }


# ─────────────────────────────────────────────
#  MARCHÉ : PARSING DES COTES + DEVIG "POWER"
# ─────────────────────────────────────────────

def _parse_odds(odds_response: list) -> tuple[dict, dict, str]:
    """
    Retourne :
      best : marché → sélection → {"cote": meilleure cote, "book": nom}
      ref  : marché → sélection → cote du bookmaker de référence (devig)
      nom du bookmaker de référence
    """
    best: dict = {}
    ref:  dict = {}
    bookmakers = odds_response[0].get("bookmakers", []) if odds_response else []
    ref_bk = None
    for pid in PREFERRED_BOOKMAKER_IDS:
        ref_bk = next((b for b in bookmakers if b.get("id") == pid), None)
        if ref_bk:
            break
    if ref_bk is None and bookmakers:
        ref_bk = bookmakers[0]

    for bk in bookmakers:
        for bet in bk.get("bets", []):
            if bet.get("name") not in MARKETS_MAP:
                continue
            mlabel, selmap = MARKETS_MAP[bet["name"]]
            for v in bet.get("values", []):
                if v.get("value") not in selmap:
                    continue
                sel = selmap[v["value"]]
                odd = _f(v.get("odd"))
                if odd <= 1.0:
                    continue
                cur = best.setdefault(mlabel, {}).get(sel)
                if cur is None or odd > cur["cote"]:
                    best[mlabel][sel] = {"cote": odd, "book": bk.get("name", "?")}
                if bk is ref_bk:
                    ref.setdefault(mlabel, {})[sel] = odd
    return best, ref, (ref_bk.get("name", "?") if ref_bk else "")


def _devig_power(odds: list[float]) -> tuple[list[float], float]:
    """
    Retire la marge bookmaker par la méthode "power" : p_i = (1/o_i)^k,
    k résolu pour que Σp = 1. Moins biaisée que la méthode multiplicative
    sur les outsiders (biais favori-outsider). Retourne (probas, marge).
    """
    imps   = [1.0 / o for o in odds]
    margin = sum(imps) - 1.0
    lo, hi = 0.5, 5.0
    for _ in range(60):
        k = (lo + hi) / 2.0
        if sum(ip ** k for ip in imps) > 1.0:
            lo = k
        else:
            hi = k
    probs = [ip ** ((lo + hi) / 2.0) for ip in imps]
    total = sum(probs)
    return [p / total for p in probs], round(margin * 100, 2)


# ─────────────────────────────────────────────
#  VALUE BETTING : BLEND + EDGE + EV + KELLY
# ─────────────────────────────────────────────

def _market_value_bets(
    market: str,
    model_probs: dict[str, float],
    ref_odds: dict[str, float],
    best_odds: dict[str, dict],
    threshold_mult: float,
    labels: dict[str, str],
) -> tuple[list[dict], dict, list[str]]:
    """
    Pour un marché donné :
      1. Devig power des cotes du bookmaker de référence → probas "justes"
      2. Blend : p_finale = 65 % marché + 35 % modèle
         (edge = 0.35 × désaccord modèle-marché → il faut un vrai désaccord)
      3. Pick si edge ≥ seuil ET EV ≥ seuil (EV sur la MEILLEURE cote)
      4. Un seul pick max par marché (le meilleur EV) — deux issues
         gagnantes du même marché = modèle mal calibré, pas double chance.
    Retourne (picks, infos marché, warnings).
    """
    sels = MARKET_SELECTIONS[market]
    warnings: list[str] = []
    if not all(s in ref_odds for s in sels):
        return [], {}, warnings

    fair_list, margin = _devig_power([ref_odds[s] for s in sels])
    fair    = dict(zip(sels, fair_list))
    blended = {s: MARKET_WEIGHT * fair[s] + (1.0 - MARKET_WEIGHT) * model_probs[s] for s in sels}

    candidates = []
    for s in sels:
        info = best_odds.get(s)
        if not info or info["cote"] <= 1.0:
            continue
        cote  = info["cote"]
        edge  = blended[s] - fair[s]
        ev    = blended[s] * cote - 1.0
        if edge >= EDGE_THRESHOLD * threshold_mult and ev >= EV_THRESHOLD * threshold_mult:
            kelly = min(max(0.0, ev / (cote - 1.0)) * KELLY_FRACTION, MAX_STAKE_PER_BET)
            candidates.append({
                "marché":       market,
                "pari":         s,
                "libellé":      labels[s],
                "cote":         cote,
                "bookmaker":    info["book"],
                "p_modèle_%":   round(model_probs[s] * 100, 1),
                "p_marché_%":   round(fair[s] * 100, 1),
                "p_finale_%":   round(blended[s] * 100, 1),
                "edge_%":       round(edge * 100, 1),
                "EV_%":         round(ev * 100, 1),
                "mise_kelly_%": round(kelly * 100, 2),
            })

    candidates.sort(key=lambda x: x["EV_%"], reverse=True)
    if len(candidates) > 1:
        warnings.append(
            f"Anomalie {market} : plusieurs issues du même marché passaient les seuils "
            f"— seul le meilleur EV est conservé (signe de calibration à surveiller)."
        )
    market_info = {
        "marge_%":  margin,
        "p_marché": {s: round(fair[s] * 100, 1) for s in sels},
        "p_finale": {s: round(blended[s] * 100, 1) for s in sels},
    }
    return candidates[:1], market_info, warnings


def _apply_exposure_cap(picks: list[dict], warnings: list[str]) -> list[dict]:
    """Plafonne l'exposition cumulée de la journée à DAILY_EXPOSURE_CAP."""
    global _RUN_EXPOSURE
    kept = []
    for pick in sorted(picks, key=lambda x: x["EV_%"], reverse=True):
        stake     = pick["mise_kelly_%"] / 100.0
        remaining = DAILY_EXPOSURE_CAP - _RUN_EXPOSURE
        if remaining < 0.0025:
            warnings.append(
                f"Plafond journalier de {DAILY_EXPOSURE_CAP:.0%} atteint — "
                f"pari {pick['libellé']} écarté malgré un EV de {pick['EV_%']}%."
            )
            continue
        if stake > remaining:
            stake = remaining
            pick["mise_kelly_%"] = round(stake * 100, 2)
            warnings.append(f"Mise réduite ({pick['libellé']}) pour respecter le plafond journalier.")
        _RUN_EXPOSURE += stake
        kept.append(pick)
    return kept


# ─────────────────────────────────────────────
#  SUIVI DES PERFORMANCES (CSV + SETTLEMENT)
# ─────────────────────────────────────────────

def _ensure_csv_schema(path: Path, fields: list[str]) -> None:
    """Si un ancien CSV avec un autre schéma existe, on l'archive."""
    if not path.exists():
        return
    with open(path, encoding="utf-8", newline="") as f:
        header = f.readline().strip()
    if header != ",".join(fields):
        legacy = path.with_name(path.stem + "_legacy.csv")
        path.rename(legacy)
        logging.info(f"[Perf] Ancien schéma CSV archivé → {legacy.name}")


def _read_history(path: Path, fields: list[str]) -> list[dict]:
    _ensure_csv_schema(path, fields)
    if not path.exists():
        return []
    with open(path, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _append_history(path: Path, fields: list[str], rows: list[dict]) -> None:
    needs_header = not path.exists()
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        if needs_header:
            w.writeheader()
        w.writerows(rows)


def _write_history(path: Path, fields: list[str], rows: list[dict]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def _log_bets_to_csv(date_str: str, match_label: str, fixture_id: int, bets: list[dict]) -> None:
    """
    Enregistre chaque value bet dans bet_history.csv.
    résultat/gain_unités sont remplis automatiquement par settle_pending_bets().
    clv reste manuelle (cotes de clôture non fiables sur le plan Free).
    Dédup sur (fixture_id, marché, pari) : un re-run ne double pas les lignes.
    """
    if not bets:
        return
    existing = {
        (r["fixture_id"], r["marché"], r["pari"])
        for r in _read_history(PERF_FILE, PERF_FIELDS)
    }
    rows = [{
        "date":        date_str,
        "fixture_id":  fixture_id,
        "match":       match_label,
        "marché":      bet["marché"],
        "pari":        bet["pari"],
        "cote":        bet["cote"],
        "bookmaker":   bet["bookmaker"],
        "p_modèle_%":  bet["p_modèle_%"],
        "p_marché_%":  bet["p_marché_%"],
        "p_finale_%":  bet["p_finale_%"],
        "edge_%":      bet["edge_%"],
        "ev_%":        bet["EV_%"],
        "mise_%":      bet["mise_kelly_%"],
        "résultat":    "",
        "gain_unités": "",
        "clv":         "",
    } for bet in bets if (str(fixture_id), bet["marché"], bet["pari"]) not in existing]
    if rows:
        _append_history(PERF_FILE, PERF_FIELDS, rows)
        logging.info(f"[Perf] {len(rows)} value bet(s) loggé(s) dans bet_history.csv")


def _log_mpp_to_csv(date_str: str, match_label: str, fixture_id: int, mpp: dict) -> None:
    """Enregistre le prono Mon Petit Prono (1 par match, dédup sur fixture_id)."""
    if not mpp:
        return
    existing = {r["fixture_id"] for r in _read_history(MPP_FILE, MPP_FIELDS)}
    if str(fixture_id) in existing:
        return
    _append_history(MPP_FILE, MPP_FIELDS, [{
        "date":            date_str,
        "fixture_id":      fixture_id,
        "match":           match_label,
        "score_conseillé": mpp["score_conseillé"],
        "points_attendus": mpp["points_attendus"],
        "score_réel":      "",
        "points":          "",
        "résultat":        "",
    }])
    logging.info(f"[Perf] Prono MPP {mpp['score_conseillé']} loggé pour {match_label}")


def _bet_won(marche: str, pari: str, gh: int, ga: int) -> bool:
    if marche == "1X2":
        actual = "1" if gh > ga else ("2" if ga > gh else "X")
        return pari == actual
    if marche == "O/U 2.5":
        return (pari == "Over") == (gh + ga >= 3)
    if marche == "BTTS":
        return (pari == "Oui") == (gh >= 1 and ga >= 1)
    return False


def _sign(x: int) -> int:
    return (x > 0) - (x < 0)


def _fetch_results(pending_ids: list[str]) -> dict[str, tuple]:
    """
    Scores des fixtures : (statut, but_90_dom, but_90_ext, but_final_dom, but_final_ext).
    - score 90 min (score.fulltime) → règlement des PARIS (règle bookmaker)
    - score final hors tirs au but (goals, prolongation incluse) → règlement MPP
    L'endpoint accepte 20 ids par requête.
    """
    results: dict[str, tuple] = {}
    for i in range(0, len(pending_ids), 20):
        batch = pending_ids[i:i + 20]
        try:
            for m in _api_get("fixtures", {"ids": "-".join(batch)}):
                status = m["fixture"]["status"]["short"]
                tot_h, tot_a = m["goals"]["home"], m["goals"]["away"]
                ft     = (m.get("score") or {}).get("fulltime") or {}
                ft_h, ft_a = ft.get("home"), ft.get("away")
                if ft_h is None or ft_a is None:
                    ft_h, ft_a = tot_h, tot_a
                if tot_h is None or tot_a is None:
                    tot_h, tot_a = ft_h, ft_a
                results[str(m["fixture"]["id"])] = (status, ft_h, ft_a, tot_h, tot_a)
        except Exception as e:
            logging.warning(f"[Perf] Settlement impossible pour le lot {batch}: {e}")
    return results


def _bet_stats(rows: list[dict]) -> str:
    """Ligne de bilan pour un sous-ensemble de paris réglés."""
    settled = [r for r in rows if r["résultat"] in ("gagné", "perdu")]
    if not settled:
        return ""
    n      = len(settled)
    wins   = sum(1 for r in settled if r["résultat"] == "gagné")
    staked = sum(_f(r["mise_%"]) for r in settled)
    profit = sum(_f(r["gain_unités"]) for r in settled)
    yield_ = (profit / staked * 100) if staked else 0.0
    brier  = sum(
        (_f(r["p_finale_%"]) / 100 - (1.0 if r["résultat"] == "gagné" else 0.0)) ** 2
        for r in settled
    ) / n
    return (
        f"{n} paris réglés — {wins} gagnés ({wins / n * 100:.0f}%) · "
        f"misé {staked:.1f}% · profit {profit:+.2f} pts de bankroll · "
        f"yield {yield_:+.1f}% · Brier {brier:.3f}"
    )


def _mpp_stats(rows: list[dict]) -> str:
    """Ligne de bilan pour un sous-ensemble de pronos MPP réglés."""
    settled = [r for r in rows if r["résultat"] in ("score exact", "bon résultat", "raté")]
    if not settled:
        return ""
    n     = len(settled)
    exact = sum(1 for r in settled if r["résultat"] == "score exact")
    good  = exact + sum(1 for r in settled if r["résultat"] == "bon résultat")
    pts   = sum(_f(r["points"]) for r in settled)
    return (
        f"{n} pronos réglés — bon résultat {good / n * 100:.0f}% · "
        f"score exact {exact / n * 100:.0f}% · {pts / n:.2f} pts/match "
        f"(barème {MPP_PTS_RESULT}/{MPP_PTS_EXACT})"
    )


def settle_pending_bets() -> str:
    """
    Règle les paris ET les pronos Mon Petit Prono en attente (scores 90 min),
    puis retourne un bilan texte : réglé ce matin + cumul, et le dimanche
    un bilan hebdomadaire (paris + MPP sur les 7 derniers jours).
    """
    bet_rows = _read_history(PERF_FILE, PERF_FIELDS)
    mpp_rows = _read_history(MPP_FILE, MPP_FIELDS)
    if not bet_rows and not mpp_rows:
        return ""

    pending_ids = sorted(
        {r["fixture_id"] for r in bet_rows if not r["résultat"]}
        | {r["fixture_id"] for r in mpp_rows if not r["résultat"]}
    )
    results = _fetch_results(pending_ids) if pending_ids else {}

    # ── Paris ─────────────────────────────────────────────────────
    settled_now, profit_now = 0, 0.0
    for r in bet_rows:
        if r["résultat"] or r["fixture_id"] not in results:
            continue
        status, gh, ga, _, _ = results[r["fixture_id"]]
        if status in ("CANC", "ABD"):
            r["résultat"], r["gain_unités"] = "annulé", "0.0"
            continue
        if status not in ("FT", "AET", "PEN") or gh is None or ga is None:
            continue  # pas encore joué / reporté → reste en attente
        # Marchés pré-match réglés sur le temps réglementaire (score fulltime)
        mise = _f(r["mise_%"])
        if _bet_won(r["marché"], r["pari"], int(gh), int(ga)):
            r["résultat"]    = "gagné"
            r["gain_unités"] = str(round(mise * (_f(r["cote"]) - 1.0), 3))
        else:
            r["résultat"]    = "perdu"
            r["gain_unités"] = str(round(-mise, 3))
        settled_now += 1
        profit_now  += _f(r["gain_unités"])

    # ── Mon Petit Prono ───────────────────────────────────────────
    for r in mpp_rows:
        if r["résultat"] or r["fixture_id"] not in results:
            continue
        # MPP se juge sur le score final hors tirs au but (120 min si prolongation)
        status, _, _, gh, ga = results[r["fixture_id"]]
        if status in ("CANC", "ABD"):
            r["résultat"] = "annulé"
            continue
        if status not in ("FT", "AET", "PEN") or gh is None or ga is None:
            continue
        gh, ga = int(gh), int(ga)
        r["score_réel"] = f"{gh}-{ga}"
        try:
            ph, pa = (int(x) for x in r["score_conseillé"].split("-"))
        except ValueError:
            r["résultat"] = "annulé"
            continue
        if (ph, pa) == (gh, ga):
            r["résultat"], r["points"] = "score exact", str(MPP_PTS_EXACT)
        elif _sign(ph - pa) == _sign(gh - ga):
            r["résultat"], r["points"] = "bon résultat", str(MPP_PTS_RESULT)
        else:
            r["résultat"], r["points"] = "raté", "0"

    if bet_rows:
        _write_history(PERF_FILE, PERF_FIELDS, bet_rows)
    if mpp_rows:
        _write_history(MPP_FILE, MPP_FIELDS, mpp_rows)

    # ── Bilan texte ───────────────────────────────────────────────
    lines: list[str] = []
    if settled_now:
        lines.append(f"Nouveaux paris réglés ce matin : {settled_now} → {profit_now:+.2f} pts de bankroll")
    if (s := _bet_stats(bet_rows)):
        lines.append(f"Paris (cumul) : {s}")
    if (s := _mpp_stats(mpp_rows)):
        lines.append(f"Mon Petit Prono (cumul) : {s}")

    # Dimanche : bilan hebdomadaire sur les 7 derniers jours
    today = _now_fr().date()
    if today.weekday() == 6:
        week_start = (today - timedelta(days=6)).isoformat()
        week_bets  = [r for r in bet_rows if r["date"] >= week_start]
        week_mpp   = [r for r in mpp_rows if r["date"] >= week_start]
        week_lines = []
        if (s := _bet_stats(week_bets)):
            week_lines.append(f"Paris : {s}")
        if (s := _mpp_stats(week_mpp)):
            week_lines.append(f"Mon Petit Prono : {s}")
        if week_lines:
            lines.append(f"BILAN HEBDO (semaine du {week_start} au {today.isoformat()}) :")
            lines.extend("  " + wl for wl in week_lines)

    return "\n".join(lines)


# ─────────────────────────────────────────────
#  OUTILS
# ─────────────────────────────────────────────

def get_fixtures_today() -> str:
    """
    Matchs du jour (heure de Paris) avec sélection DÉTERMINISTE en Python :
    tous les matchs de Coupe du Monde, puis max 3/ligue et 15 au total
    (budget API : 2 requêtes par match analysé).
    """
    today = _now_fr().date().isoformat()
    # timezone=Europe/Paris : sinon l'API filtre sur le jour UTC et les
    # matchs de fin de soirée basculent sur la mauvaise date.
    all_fix = _api_get("fixtures", {"date": today, "timezone": "Europe/Paris"})

    candidates: list[dict] = []
    for m in all_fix:
        lid = m["league"]["id"]
        if lid not in TARGET_LEAGUES:
            continue
        if m["fixture"]["status"]["short"] not in ("NS", "TBD"):
            continue  # déjà joué / en cours → inutile de pronostiquer
        round_name = (m["league"].get("round") or "").lower()
        candidates.append({
            "fixture_id":       m["fixture"]["id"],
            "tournament":       TARGET_LEAGUES[lid],
            "league_id":        lid,
            "is_neutral_venue": lid in NEUTRAL_LEAGUES,
            # Élimination directe (WC hors phase de groupes) : prolongation
            # possible → Mon Petit Prono se joue sur le score à 120 min
            "is_knockout":      lid in NEUTRAL_LEAGUES and "group" not in round_name,
            "heure_fr":         m["fixture"]["date"][11:16],
            "home_team":        m["teams"]["home"]["name"],
            "home_id":          m["teams"]["home"]["id"],
            "away_team":        m["teams"]["away"]["name"],
            "away_id":          m["teams"]["away"]["id"],
        })

    # Sélection : WC d'abord (tous), puis ligues dans l'ordre de TARGET_LEAGUES
    selected: list[dict] = [c for c in candidates if c["league_id"] in NEUTRAL_LEAGUES]
    per_league: dict[int, int] = {}
    for lid in TARGET_LEAGUES:
        if lid in NEUTRAL_LEAGUES:
            continue
        for c in candidates:
            if len(selected) >= MAX_MATCHES_PER_DAY:
                break
            if c["league_id"] == lid and per_league.get(lid, 0) < MAX_MATCHES_PER_LEAGUE:
                selected.append(c)
                per_league[lid] = per_league.get(lid, 0) + 1
    selected = selected[:MAX_MATCHES_PER_DAY]
    selected.sort(key=lambda c: c["heure_fr"])

    if not selected:
        return "Aucun match trouvé aujourd'hui pour les ligues sélectionnées."
    return json.dumps({
        "nb_matchs_sélectionnés": len(selected),
        "consigne": "Analyse TOUS ces matchs (la sélection est déjà faite).",
        "matchs": selected,
    }, ensure_ascii=False, indent=2)


def get_match_analysis(
    fixture_id: int,
    is_neutral_venue: bool,
    league_id: int,
    home_team: str = "",
    away_team: str = "",
    is_knockout: bool = False,
) -> str:
    """
    Analyse complète orientée value betting :
    1. Stats saison (splits dom/ext) + forme récente via /predictions
    2. Poisson forces relatives + shrinkage + Dixon-Coles → matrice des scores
    3. Marchés 1X2, O/U 2.5, BTTS : devig power (book de référence),
       blend 65 % marché / 35 % modèle, picks si edge/EV suffisants
    4. Mise Kelly/4 plafonnée (2 %/pari, 8 %/jour)
    5. Score conseillé Mon Petit Prono
    6. Avertissements QA
    """
    threshold_mult = WC_THRESHOLD_MULT if is_neutral_venue else 1.0
    warnings: list[str] = []

    # ── 1. Prédictions API (stats saison + forme + H2H) ───────────
    preds = _api_get("predictions", {"fixture": fixture_id})
    forme: dict = {}
    h2h_list: list = []
    model_block: dict = {}
    mpp: dict = {}
    elo_block: dict = {}
    matrix = None
    mk: dict = {}
    home_s = away_s = None

    if preds:
        p      = preds[0]
        teams  = p.get("teams", {})
        home_s = _team_stats(teams.get("home") or {})
        away_s = _team_stats(teams.get("away") or {})

        forme = {
            "domicile": {
                "saison": {
                    "matchs":                home_s["n_total"],
                    "buts_marqués_dom":      home_s["gf_home"],
                    "buts_encaissés_dom":    home_s["ga_home"],
                    "buts_marqués_total":    home_s["gf_total"],
                    "buts_encaissés_total":  home_s["ga_total"],
                },
                "5_derniers": {
                    "matchs":          home_s["l5_played"],
                    "buts_marqués":    home_s["l5_gf"],
                    "buts_encaissés":  home_s["l5_ga"],
                    "forme_%":         home_s["l5_forme"],
                },
            },
            "extérieur": {
                "saison": {
                    "matchs":                away_s["n_total"],
                    "buts_marqués_ext":      away_s["gf_away"],
                    "buts_encaissés_ext":    away_s["ga_away"],
                    "buts_marqués_total":    away_s["gf_total"],
                    "buts_encaissés_total":  away_s["ga_total"],
                },
                "5_derniers": {
                    "matchs":          away_s["l5_played"],
                    "buts_marqués":    away_s["l5_gf"],
                    "buts_encaissés":  away_s["l5_ga"],
                    "forme_%":         away_s["l5_forme"],
                },
            },
        }

        # H2H : 3 derniers matchs — indicateur faible, jamais un argument principal
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

        lam_h, lam_a, strengths = _expected_goals(home_s, away_s, league_id, is_neutral_venue)

        # Matchs internationaux : les stats sont bruitées → ancrage sur l'Elo
        if league_id in NEUTRAL_LEAGUES:
            lam_h, lam_a, elo_block, elo_warns = _apply_elo(
                lam_h, lam_a, home_team, away_team, league_id
            )
            warnings.extend(elo_warns)

        matrix = _score_matrix(lam_h, lam_a)
        mk     = _matrix_markets(matrix)

        model_block = {
            "λ_dom":               lam_h,
            "λ_ext":               lam_a,
            "buts_attendus_total": round(lam_h + lam_a, 2),
            "p_victoire_dom_%":    round(mk["p1"] * 100, 1),
            "p_nul_%":             round(mk["px"] * 100, 1),
            "p_victoire_ext_%":    round(mk["p2"] * 100, 1),
            "p_over_2.5_%":        round(mk["over25"] * 100, 1),
            "p_BTTS_%":            round(mk["btts"] * 100, 1),
            "top_scores":          mk["top_scores"],
            "forces":              strengths,
        }
    else:
        warnings.append("Prédictions API indisponibles — aucune analyse modèle possible.")

    # ── 2. Cotes (tous bookmakers) ────────────────────────────────
    best_odds: dict = {}
    ref_odds:  dict = {}
    ref_name = ""
    try:
        odds_list = _api_get("odds", {"fixture": fixture_id})
        best_odds, ref_odds, ref_name = _parse_odds(odds_list)
    except Exception as e:
        logging.warning(f"Erreur cotes fixture {fixture_id}: {e}")

    # ── 3. Value betting multi-marchés ────────────────────────────
    value_bets: list[dict] = []
    market_info: dict = {}
    if mk and ref_odds:
        home_lbl = home_team or "domicile"
        away_lbl = away_team or "extérieur"
        market_defs = [
            ("1X2",
             {"1": mk["p1"], "X": mk["px"], "2": mk["p2"]},
             {"1": f"Victoire {home_lbl}", "X": "Match nul", "2": f"Victoire {away_lbl}"}),
            ("O/U 2.5",
             {"Over": mk["over25"], "Under": 1.0 - mk["over25"]},
             {"Over": "Plus de 2,5 buts", "Under": "Moins de 2,5 buts"}),
            ("BTTS",
             {"Oui": mk["btts"], "Non": 1.0 - mk["btts"]},
             {"Oui": "Les deux équipes marquent", "Non": "Au moins une équipe ne marque pas"}),
        ]
        for market, probs, labels in market_defs:
            picks, info, w = _market_value_bets(
                market, probs, ref_odds.get(market, {}), best_odds.get(market, {}),
                threshold_mult, labels,
            )
            value_bets.extend(picks)
            warnings.extend(w)
            if info:
                market_info[market] = info
        value_bets = _apply_exposure_cap(value_bets, warnings)
        value_bets.sort(key=lambda x: x["EV_%"], reverse=True)
    elif mk and not ref_odds:
        warnings.append("Cotes non disponibles — calcul EV impossible, aucun value bet.")

    # ── 4. Mon Petit Prono ────────────────────────────────────────
    # NB : les paris (1X2, O/U, BTTS) restent réglés sur 90 minutes (règle
    # bookmaker standard) ; seul MPP se joue sur 120 min en élimination directe.
    if matrix is not None:
        blend_1x2 = (market_info.get("1X2") or {}).get("p_finale")
        if blend_1x2:
            p1b, pxb, p2b = blend_1x2["1"] / 100, blend_1x2["X"] / 100, blend_1x2["2"] / 100
        else:
            p1b, pxb, p2b = mk["p1"], mk["px"], mk["p2"]
        if is_knockout:
            # MPP juge sur le score APRÈS prolongation : un nul à 90' peut
            # devenir 2-1, et un nul à 120' (→ tirs au but) reste un nul.
            m120, q1, qx, q2 = _extend_matrix_120(matrix, lam_h, lam_a)
            mpp = _mon_petit_prono(
                m120,
                p1b + pxb * q1,   # gagne dans le temps réglementaire OU en prolongation
                pxb * qx,         # toujours nul après 120 min (tirs au but)
                p2b + pxb * q2,
            )
            mpp["décompte"] = "score après prolongation (120 min) — nul = décision aux tirs au but"
        else:
            mpp = _mon_petit_prono(matrix, p1b, pxb, p2b)

    # ── 5. QA ─────────────────────────────────────────────────────
    if home_s and (home_s["n_total"] < 5 or away_s["n_total"] < 5):
        warnings.append(
            f"Petit échantillon saison ({home_s['n_total']}/{away_s['n_total']} matchs) "
            "— estimation fortement régularisée vers la moyenne de ligue."
        )
    if is_neutral_venue:
        if elo_block:
            warnings.append(
                "Match international : modèle ancré sur le classement Elo mondial "
                "(70 %), stats récentes en appoint — seuils de value doublés par prudence."
            )
        else:
            warnings.append(
                "Match international : les stats mélangent des adversaires de niveaux très "
                "différents — seuils de value doublés, confiance réduite."
            )

    # Log automatique pour suivi performances (paris + prono MPP)
    match_label = f"{home_team} vs {away_team}" if home_team else str(fixture_id)
    today_iso   = _now_fr().date().isoformat()
    _log_bets_to_csv(today_iso, match_label, fixture_id, value_bets)
    _log_mpp_to_csv(today_iso, match_label, fixture_id, mpp)

    return json.dumps({
        "forme":                forme,
        "h2h":                  h2h_list,
        "elo":                  elo_block,
        "modèle":               model_block,
        "bookmaker_référence":  ref_name,
        "marchés":              market_info,
        "value_bets":           value_bets,
        "mon_petit_prono":      mpp,
        "avertissements":       warnings,
        "exposition_cumulée_%": round(_RUN_EXPOSURE * 100, 2),
    }, ensure_ascii=False, indent=2)


def _split_message(text: str, limit: int = 4000) -> list[str]:
    """Découpe aux sauts de ligne : ne coupe jamais une balise HTML en deux."""
    parts, current = [], ""
    for line in text.split("\n"):
        while len(line) > limit:
            parts.append(line[:limit])
            line = line[limit:]
        if len(current) + len(line) + 1 > limit:
            parts.append(current)
            current = line
        else:
            current = f"{current}\n{line}" if current else line
    if current:
        parts.append(current)
    return parts


def send_telegram_report(text: str) -> str:
    url    = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    chunks = _split_message(text)
    for chunk in chunks:
        resp = requests.post(
            url,
            json={"chat_id": TELEGRAM_CHAT_ID, "text": chunk, "parse_mode": "HTML"},
            timeout=15,
        )
        if resp.status_code == 400:
            # HTML invalide → fallback texte brut plutôt que perdre le rapport
            logging.warning("Telegram a refusé le HTML — envoi en texte brut.")
            resp = requests.post(
                url,
                json={"chat_id": TELEGRAM_CHAT_ID, "text": re.sub(r"<[^>]+>", "", chunk)},
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
            "Récupère les matchs du jour DÉJÀ SÉLECTIONNÉS (Coupe du Monde en entier, "
            "puis max 3/ligue parmi Premier League, La Liga, Bundesliga, Serie A, Ligue 1). "
            "Retourne fixture_id, league_id, équipes + IDs, heure FR, is_neutral_venue. "
            "Toujours appeler EN PREMIER, sans argument. Analyser TOUS les matchs retournés."
        ),
        "input_schema": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "get_match_analysis",
        "description": (
            "Analyse complète orientée value betting. Retourne : forme (saison avec splits "
            "domicile/extérieur + 5 derniers matchs), elo (matchs internationaux : classement "
            "Elo mondial des deux équipes, base principale du modèle), H2H (faible poids prédictif), "
            "probabilités du modèle (1X2, Over/Under 2.5, BTTS, scores les plus probables), "
            "marchés déviggés, probabilités finales (blend marché/modèle), value_bets, "
            "et mon_petit_prono (score conseillé + points attendus). "
            "value_bets = paris où l'estimation finale bat le marché (edge et EV suffisants). "
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
                "is_knockout": {
                    "type": "boolean",
                    "description": (
                        "true pour un match à élimination directe (fourni par "
                        "get_fixtures_today) — Mon Petit Prono se joue alors sur 120 min"
                    ),
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
                    is_knockout      = tool_input.get("is_knockout", False),
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
    today = _now_fr().strftime("%d/%m/%Y")
    return f"""Tu es un analyste sportif qui rédige des rapports de pronostics accessibles à tous.
Date : {today}.

Le moteur de décision probabiliste tourne en Python. Ton rôle : orchestrer les appels d'outils,
puis rédiger un rapport précis et détaillé que n'importe qui peut comprendre — même sans
connaître les statistiques.

━━━━━━━━━━━━━━━━━━━━━━━━
ÉTAPES OBLIGATOIRES
━━━━━━━━━━━━━━━━━━━━━━━━
1. Appelle get_fixtures_today (sans argument). La sélection des matchs est déjà faite.
2. Si aucun match → envoie un Telegram le signalant et arrête.
3. Pour CHAQUE match retourné : appelle get_match_analysis avec fixture_id, is_neutral_venue,
   is_knockout, league_id, home_team et away_team.
4. Génère le rapport et appelle send_telegram_report UNE SEULE FOIS.

━━━━━━━━━━━━━━━━━━━━━━━━
RÈGLE CENTRALE — VALUE BETTING
━━━━━━━━━━━━━━━━━━━━━━━━
- Les probabilités FINALES (celles à afficher) combinent déjà le marché (65 %) et notre
  modèle (35 %) : quand un pari sort, c'est que le désaccord avec les bookmakers survit
  même après cette prudence.
- Si value_bets est vide → affiche "Aucun pari recommandé" — N'INVENTE PAS de pick.
- N'utilise PAS le H2H comme argument principal (effectifs changent, trop peu de matchs).
- La mise recommandée est pré-calculée (Kelly/4, plafonnée à 2 % par pari et 8 % par jour)
  — utilise-la telle quelle.
- Tu expliques les données, tu ne prends pas de décision : le modèle l'a déjà fait.
- Les paris peuvent porter sur 3 marchés : 1X2, Plus/Moins de 2,5 buts, Les deux équipes
  marquent. Traduis toujours en français clair (jamais "Over 2.5" ou "BTTS" seuls).

━━━━━━━━━━━━━━━━━━━━━━━━
FORMAT DU RAPPORT (HTML strict, Telegram)
━━━━━━━━━━━━━━━━━━━━━━━━

<b>⚽ PRONOSTICS DU {today}</b>
<i>Analyse statistique indépendante des bookmakers</i>

[Si un bilan de performance est fourni dans le message initial :]
📈 <b>BILAN DU MODÈLE</b>
[Recopie fidèlement les chiffres fournis, en lignes claires :
 • Paris : nombre réglé, % gagnés, profit, yield (si le yield est négatif, dis-le honnêtement)
 • Mon Petit Prono : % de bons résultats, % de scores exacts, points par match
 N'affiche PAS le score de Brier (indicateur interne).
 Si une partie "BILAN HEBDO" est fournie (le dimanche), présente-la dans un sous-bloc
 distinct "📅 <b>Bilan de la semaine</b>" avec les mêmes règles.]

━━━━━━━━━━━━━━━━━━━━━━━━

[Si matchs WC présents]
<b>🌍 FIFA COUPE DU MONDE 2026</b>

[Si matchs de ligues européennes]
<b>🏆 LIGUES EUROPÉENNES</b>

─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─

Pour CHAQUE match, reproduis exactement cette structure :

<b>🏟 [Éq. A] vs [Éq. B]</b>
<i>[Compétition] — 🕐 [Heure]</i>

📊 <b>Forme :</b>
  • [Éq. A] : saison à domicile [x] buts marqués / [x] encaissés par match — 5 derniers : [x] marqués, [x] encaissés
  • [Éq. B] : saison à l'extérieur [x] buts marqués / [x] encaissés par match — 5 derniers : [x] marqués, [x] encaissés
  [Sur terrain neutre (WC) : utilise les stats "total" saison, sans mention domicile/extérieur]

[Si le champ "elo" est présent (matchs internationaux) :]
🌍 <b>Classement Elo mondial :</b> [Éq. A] [rang]e ([elo] pts) · [Éq. B] [rang]e ([elo] pts)
[Si bonus_hôte_appliqué est true, ajoute : <i>(avantage du pays hôte pris en compte)</i>]

🔢 <b>Notre estimation finale :</b>
  • Victoire [Éq. A] : <b>[X]%</b>   ← bookmakers donnent [X]%
  • Match nul : <b>[X]%</b>           ← bookmakers donnent [X]%
  • Victoire [Éq. B] : <b>[X]%</b>   ← bookmakers donnent [X]%
  ⚽ Buts attendus : [X.X] au total · Plus de 2,5 buts : [X]% · Les deux marquent : [X]%
  <i>Score le plus probable : [X-X] ([Y]% de chances)</i>
  [Utilise p_finale des marchés pour "estimation finale" et p_marché pour "bookmakers".
   Si les cotes manquent, utilise les probabilités du modèle et signale-le.]

📝 <b>Analyse :</b>
[OBLIGATOIRE — 4 à 6 phrases en langage naturel, précises et chiffrées. Tu dois expliquer :
 1. Quelle équipe est la plus solide sur la SAISON (avec les splits domicile/extérieur)
    et laquelle arrive en meilleure forme récente — cite les chiffres.
    Pour les matchs internationaux : le classement Elo mondial est l'argument de force
    PRINCIPAL (notre modèle s'appuie dessus à 70 %), les stats récentes viennent en appoint.
    Mentionne l'avantage du pays hôte s'il s'applique.
 2. Le profil de buts attendu : match ouvert ou fermé ? (buts attendus, % over 2.5).
 3. Ce que voit notre modèle : y a-t-il un écart avec les bookmakers ? Sur quelle issue,
    et pourquoi c'est potentiellement intéressant.
 4. Conclusion claire : le match vaut-il un pari, et sur quel marché ?
 Ton ton doit être celui d'un ami qui explique le match simplement, sans jargon technique.
 Exemples de formulations :
 - "À domicile, Dortmund marque 2,4 buts par match cette saison, et sa forme récente confirme."
 - "Notre estimation donne 31% au Japon contre 24% pour les bookmakers — c'est là qu'on voit de la valeur."
 - "Avec 3,1 buts attendus, tout indique un match ouvert : 62% de chances de voir plus de 2,5 buts."
 - "Aucun écart significatif avec les cotes : pas d'avantage à parier ici."]

[Si value_bets non vide :]
💰 <b>Pari recommandé :</b>
[Pour chaque value bet :]
  ✅ <b>[libellé du pari]</b> @ cote [X.XX] (chez [bookmaker])
  Notre estimation : [p_finale]% · Bookmakers : [p_marché]% · Avantage : +[edge] points
  Mise conseillée : [mise_kelly_%]% de ta bankroll

[Si value_bets vide :]
🚫 <b>Aucun pari recommandé sur ce match</b>
<i>Notre estimation est trop proche des cotes du marché pour justifier un risque.</i>

🎯 <b>Mon Petit Prono :</b> [score_conseillé] <i>([points_attendus] pts attendus — alternative : [score alternatif])</i>
[Si le prono contient "décompte" (élimination directe) : ajoute
 <i>Score prolongation incluse — un nul = décision aux tirs au but.</i>]

[Si avertissements présents :]
<i>ℹ️ [Reformule l'avertissement en clair, une ligne.]</i>

━━━━━━━━━━━━━━━━━━━━━━━━

<b>📋 RÉSUMÉ DES PARIS DU JOUR</b>
[Tous les value bets de la journée, triés par avantage décroissant.
 Si aucun → "Aucun pari recommandé aujourd'hui — le marché est bien calibré sur l'ensemble
 des matchs analysés."]

1. [Match] — [libellé] @ [cote] · Estimation : [X]% vs Bookmakers : [X]% · Mise : [X]%
…
<b>Exposition totale du jour : [somme des mises]% de bankroll</b> (plafond : 8%)

<b>🎯 RÉCAP MON PETIT PRONO</b>
[Un score par match analysé, dans l'ordre chronologique :]
• [Éq. A] - [Éq. B] → <b>[score_conseillé]</b>

<i>⚠️ Ces analyses sont à titre informatif. Ne pariez que ce que vous pouvez vous permettre de perdre.</i>

━━━━━━━━━━━━━━━━━━━━━━━━
CONSIGNES DE RÉDACTION
━━━━━━━━━━━━━━━━━━━━━━━━
- heure_fr est déjà en heure française — inutile de la convertir
- Terrain neutre (WC) : utilise les noms des équipes, jamais "domicile"/"extérieur"
- En Coupe du Monde à élimination directe : précise que nos probabilités portent sur
  le temps réglementaire (90 minutes)
- Évite tout jargon : pas de "lambda", "shrinkage", "EV", "edge", "blend", "devig",
  "Dixon-Coles", "Kelly" dans le rapport final. Exception : "classement Elo mondial"
  est autorisé (compréhensible par tous, comme un classement FIFA)
- Le paragraphe Analyse est OBLIGATOIRE pour chaque match — c'est la valeur ajoutée principale
- Sois direct et factuel, évite les formulations vagues ("match équilibré", "tout est possible")
- Reste sous ~1000 mots de rapport pour tenir dans les limites Telegram
"""


# ─────────────────────────────────────────────
#  BOUCLE DE L'AGENT
# ─────────────────────────────────────────────

def run_agent(max_steps: int = 60) -> None:
    global _RUN_EXPOSURE
    _RUN_EXPOSURE = 0.0

    today_str     = _now_fr().strftime("%d/%m/%Y")
    system_prompt = _build_system_prompt()

    logging.info("=" * 55)
    logging.info(f"AGENT FOOT — VALUE BETTING — {today_str}")
    logging.info("=" * 55)

    # Settlement des paris en attente AVANT l'analyse du jour
    try:
        perf_summary = settle_pending_bets()
    except Exception as e:
        logging.warning(f"[Perf] Settlement échoué : {e}")
        perf_summary = ""
    if perf_summary:
        logging.info(f"[Perf]\n{perf_summary}")

    user_msg = f"Analyse les matchs de football du {today_str} et envoie le rapport complet sur Telegram."
    if perf_summary:
        user_msg += (
            "\n\nBilan de performance du modèle (à intégrer dans la section BILAN DU MODÈLE) :\n"
            + perf_summary
        )

    messages = [{"role": "user", "content": user_msg}]

    for step in range(max_steps):
        logging.info(f"--- Tour {step + 1} ---")

        response = client.messages.create(
            model=MODEL,
            max_tokens=16384,
            system=system_prompt,
            tools=TOOLS,
            messages=messages,
        )

        for block in response.content:
            if block.type == "text" and block.text.strip():
                logging.info(f"[Claude] {block.text}")

        if response.stop_reason == "max_tokens":
            logging.error("❌ Réponse tronquée (max_tokens) — rapport potentiellement incomplet.")
            return

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

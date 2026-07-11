"""
Configuration for the World Cup 2026 Prediction System.
Update WC_2026_GROUPS with the official draw once confirmed.
"""

# ── Data ──────────────────────────────────────────────────────────────────────
DATA_URL = "https://raw.githubusercontent.com/martj42/international_results/master/results.csv"
MIN_YEAR = 2000          # Ignore matches before this year (old data is less relevant)

# ── ELO ───────────────────────────────────────────────────────────────────────
ELO_INITIAL   = 1500     # Starting ELO for any team with no history
ELO_K_NORMAL  = 20       # K-factor for friendlies / qualifiers
ELO_K_WC      = 60       # K-factor for World Cup matches (higher = more weight)

# ── Features ──────────────────────────────────────────────────────────────────
FORM_GAMES    = 10       # Recent games window for form calculation
GOALS_WINDOW  = 10       # Rolling window for goals scored/conceded averages

# ── Training ──────────────────────────────────────────────────────────────────
TRAIN_END     = "2026-06-10"   # Train on all data up to day before WC 2026 kickoff
TEST_START    = "2022-01-01"   # Evaluate on 2022+ period (includes 2022 WC + 2026 WC)
RANDOM_STATE  = 42

# ── Simulation ────────────────────────────────────────────────────────────────
N_SIMULATIONS = 10_000   # Monte Carlo runs

# ── 2026 World Cup Groups — OFFICIAL DRAW ────────────────────────────────────
# Source: https://www.fifa.com/en/tournaments/mens/worldcup/canadamexicousa2026
# Names map to the dataset (data/results.csv).  FIFA display names in comments.
WC_2026_GROUPS = {
    "A": ["Mexico", "South Korea", "Czech Republic", "South Africa"],
    # FIFA: Mexico / Korea Republic / Czechia / South Africa
    "B": ["Canada", "Bosnia and Herzegovina", "Switzerland", "Qatar"],
    # FIFA: Canada / Bosnia and Herzegovina / Switzerland / Qatar
    "C": ["Haiti", "Scotland", "Brazil", "Morocco"],
    # FIFA: Haiti / Scotland / Brazil / Morocco
    "D": ["United States", "Paraguay", "Australia", "Turkey"],
    # FIFA: USA / Paraguay / Australia / Türkiye
    "E": ["Ivory Coast", "Germany", "Ecuador", "Curacao"],
    # FIFA: Côte d'Ivoire / Germany / Ecuador / Curaçao
    "F": ["Netherlands", "Japan", "Sweden", "Tunisia"],
    # FIFA: Netherlands / Japan / Sweden / Tunisia
    "G": ["Iran", "Belgium", "Egypt", "New Zealand"],
    # FIFA: IR Iran / Belgium / Egypt / New Zealand
    "H": ["Saudi Arabia", "Spain", "Uruguay", "Cabo Verde"],
    # FIFA: Saudi Arabia / Spain / Uruguay / Cabo Verde
    "I": ["France", "Senegal", "Iraq", "Norway"],
    # FIFA: France / Senegal / Iraq / Norway
    "J": ["Argentina", "Algeria", "Austria", "Jordan"],
    # FIFA: Argentina / Algeria / Austria / Jordan
    "K": ["Portugal", "Uzbekistan", "Colombia", "Congo DR"],
    # FIFA: Portugal / Uzbekistan / Colombia / Congo DR
    "L": ["Ghana", "England", "Croatia", "Panama"],
    # FIFA: Ghana / England / Croatia / Panama
}

# Display names for teams not matching FIFA spelling exactly
FIFA_DISPLAY_NAMES = {
    "South Korea":           "Korea Republic",
    "Czech Republic":        "Czechia",
    "Turkey":                "Türkiye",
    "Ivory Coast":           "Côte d'Ivoire",
    "Curacao":               "Curaçao",
    "United States":         "USA",
    "Iran":                  "IR Iran",
    "Congo DR":              "Congo DR",
    "Bosnia and Herzegovina":"Bosnia and Herzegovina",
}

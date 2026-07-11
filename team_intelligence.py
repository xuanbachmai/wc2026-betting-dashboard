"""
team_intelligence.py — Static scouting data for all 48 WC 2026 teams.

Four signal categories (all scaled 0–10):
  1. PLAYER QUALITY   — star_rating (best player ceiling) + star_form (current form)
  2. SQUAD DEPTH      — squad_depth (how dangerous top-to-bottom)
  3. TACTICS          — tactical_score (coaching quality) + style_score (attack vs defence)
  4. POLITICAL/MOTIVATION — political_boost (host nation, pride, first WC, etc.)
                           carrying_factor (how much one player must do everything)

Notes:
  - carrying_factor penalises: high value = team collapses if star is off.
  - political_boost is a genuine edge for co-hosts (USA/Mexico/Canada) and
    for "first WC" nations (Indonesia, Venezuela) who over-perform expectations.
  - style_score: 1.0 = pure attacking/tiki-taka, 0.3 = deep defensive block.
    Used to model goal expectation biases in Poisson prediction.

Run `python team_intelligence.py` to print the full scouting table.
"""

from __future__ import annotations
import pandas as pd

# ── Tactical style → numeric ───────────────────────────────────────────────────
STYLE_ENCODING: dict[str, float] = {
    "attacking":      1.0,
    "tiki-taka":      0.9,
    "possession":     0.8,
    "high-press":     0.7,
    "counter-attack": 0.5,
    "defensive":      0.3,
}

# ── Per-team scouting data ─────────────────────────────────────────────────────
# Names MUST match data/results.csv exactly.
# Run `python main.py --list-teams` to verify.
TEAM_INTEL: dict[str, dict] = {

    # ── Group A ───────────────────────────────────────────────────────────────
    "United States": {
        "key_players":    ["Pulisic", "Reyna", "Musah", "Turner"],
        "scouting_notes": "Co-host with massive crowd advantage. Pulisic leads but no elite difference-maker.",
        "star_rating":    7.8,
        "star_form":      7.5,
        "squad_depth":    7.5,
        "tactical_score": 7.0,
        "political_boost":9.0,   # Co-host nation
        "carrying_factor":5.0,
        "style":          "high-press",
    },
    "Panama": {
        "key_players":    ["Davis", "Fajardo"],
        "scouting_notes": "Organised defensive unit. Limited attacking threat without Davis.",
        "star_rating":    5.5,
        "star_form":      5.5,
        "squad_depth":    5.0,
        "tactical_score": 5.5,
        "political_boost":5.0,
        "carrying_factor":6.0,
        "style":          "defensive",
    },
    "Honduras": {
        "key_players":    ["Elis", "Bengtson"],
        "scouting_notes": "Compact mid-block. Rely on set pieces and transitions.",
        "star_rating":    5.0,
        "star_form":      5.0,
        "squad_depth":    4.5,
        "tactical_score": 5.0,
        "political_boost":4.5,
        "carrying_factor":6.5,
        "style":          "counter-attack",
    },
    "Jamaica": {
        "key_players":    ["Antonio", "Bailey"],
        "scouting_notes": "Bailey (Bayern) is genuine threat. Physical and direct.",
        "star_rating":    6.0,
        "star_form":      5.5,
        "squad_depth":    5.5,
        "tactical_score": 5.5,
        "political_boost":5.0,
        "carrying_factor":6.5,
        "style":          "counter-attack",
    },

    # ── Group B ───────────────────────────────────────────────────────────────
    "Mexico": {
        "key_players":    ["Lozano", "Raul Jimenez", "Edson Alvarez"],
        "scouting_notes": "Co-host. Generational shift ongoing but strong base. Alvarez world class.",
        "star_rating":    7.5,
        "star_form":      7.0,
        "squad_depth":    7.0,
        "tactical_score": 7.0,
        "political_boost":8.5,   # Co-host
        "carrying_factor":5.5,
        "style":          "possession",
    },
    "Canada": {
        "key_players":    ["Davies", "David", "Buchanan"],
        "scouting_notes": "Co-host. Davies (Bayern) generational talent. David prolific scorer in Europe.",
        "star_rating":    8.0,
        "star_form":      7.5,
        "squad_depth":    7.0,
        "tactical_score": 7.0,
        "political_boost":8.0,   # Co-host
        "carrying_factor":7.0,   # Davies must do everything
        "style":          "high-press",
    },
    "Ecuador": {
        "key_players":    ["Caicedo", "Valencia", "Plata"],
        "scouting_notes": "Caicedo (Chelsea) elite midfielder. Valencia physical striker.",
        "star_rating":    7.5,
        "star_form":      7.5,
        "squad_depth":    6.5,
        "tactical_score": 6.5,
        "political_boost":5.0,
        "carrying_factor":6.5,
        "style":          "counter-attack",
    },
    "Venezuela": {
        "key_players":    ["Soteldo", "Rondon"],
        "scouting_notes": "First WC appearance — huge national pride and emotional boost.",
        "star_rating":    6.0,
        "star_form":      5.5,
        "squad_depth":    5.5,
        "tactical_score": 5.5,
        "political_boost":7.0,   # First WC
        "carrying_factor":6.0,
        "style":          "defensive",
    },

    # ── Group C ───────────────────────────────────────────────────────────────
    "Brazil": {
        "key_players":    ["Vinicius Jr", "Rodrygo", "Endrick", "Casemiro"],
        "scouting_notes": "Vini Jr arguably world's best. Deep squad across all positions.",
        "star_rating":    9.5,
        "star_form":      9.0,
        "squad_depth":    9.0,
        "tactical_score": 8.0,
        "political_boost":5.5,
        "carrying_factor":7.0,
        "style":          "attacking",
    },
    "Paraguay": {
        "key_players":    ["Alonso", "Enciso"],
        "scouting_notes": "Gritty and organised. Enciso (Brighton) is future star.",
        "star_rating":    5.5,
        "star_form":      5.5,
        "squad_depth":    5.5,
        "tactical_score": 5.5,
        "political_boost":5.0,
        "carrying_factor":5.5,
        "style":          "defensive",
    },
    "Colombia": {
        "key_players":    ["Luis Diaz", "James Rodriguez", "Jhon Duran"],
        "scouting_notes": "Luis Diaz (Liverpool) world class. James ageing but creative. Strong collective.",
        "star_rating":    8.0,
        "star_form":      8.0,
        "squad_depth":    7.5,
        "tactical_score": 7.5,
        "political_boost":5.5,
        "carrying_factor":6.5,
        "style":          "attacking",
    },
    "Bolivia": {
        "key_players":    ["Marcelo Moreno"],
        "scouting_notes": "Altitude advantage at home but very weak away. Physically limited squad.",
        "star_rating":    4.5,
        "star_form":      4.5,
        "squad_depth":    4.0,
        "tactical_score": 4.5,
        "political_boost":5.0,
        "carrying_factor":7.0,
        "style":          "defensive",
    },

    # ── Group D ───────────────────────────────────────────────────────────────
    "Argentina": {
        "key_players":    ["Messi", "De Paul", "Martinez", "Mac Allister"],
        "scouting_notes": "Messi 38-39 by WC, slight decline but still decisive. Scaloni tactically elite.",
        "star_rating":    9.8,
        "star_form":      8.5,   # Messi slowing but still produces
        "squad_depth":    9.0,
        "tactical_score": 9.0,
        "political_boost":6.0,
        "carrying_factor":8.5,   # Everything runs through Messi
        "style":          "possession",
    },
    "Uruguay": {
        "key_players":    ["Valverde", "Nunez", "Bentancur"],
        "scouting_notes": "Valverde (Real Madrid) arguably world's best box-to-box. Nunez clinical. WC2026: 66.8% poss vs Saudi Arabia, 14 corners — dominant attacking possession style.",
        "star_rating":    8.0,
        "star_form":      8.0,
        "squad_depth":    7.5,
        "tactical_score": 7.5,
        "political_boost":5.5,
        "carrying_factor":6.0,
        "style":          "attacking",   # WC2026 confirmed: dominates possession, attacks wide, high corners
    },
    "Chile": {
        "key_players":    ["Alexis Sanchez", "Ben Brereton Diaz"],
        "scouting_notes": "Golden generation faded. Rebuilding but lacking elite talent.",
        "star_rating":    6.0,
        "star_form":      5.5,
        "squad_depth":    6.0,
        "tactical_score": 6.5,
        "political_boost":5.0,
        "carrying_factor":6.5,
        "style":          "high-press",
    },
    "Peru": {
        "key_players":    ["Guerrero", "Flores"],
        "scouting_notes": "Aging squad. Guerrero likely final tournament at 42. Tactical but limited.",
        "star_rating":    5.5,
        "star_form":      5.0,
        "squad_depth":    5.5,
        "tactical_score": 5.5,
        "political_boost":5.0,
        "carrying_factor":6.0,
        "style":          "defensive",
    },

    # ── Group E ───────────────────────────────────────────────────────────────
    "England": {
        "key_players":    ["Bellingham", "Saka", "Foden", "Kane"],
        "scouting_notes": "Bellingham world class. Kane prolific. Saka/Foden elite. Huge squad depth.",
        "star_rating":    9.2,
        "star_form":      8.5,
        "squad_depth":    9.0,
        "tactical_score": 7.5,
        "political_boost":5.5,
        "carrying_factor":6.0,
        "style":          "possession",
    },
    "Ireland Republic": {
        "key_players":    ["Ogbene", "Duffy"],
        "scouting_notes": "Hard-working unit. Lack of elite creative talent in final third.",
        "star_rating":    5.0,
        "star_form":      5.0,
        "squad_depth":    5.0,
        "tactical_score": 5.5,
        "political_boost":5.5,
        "carrying_factor":5.5,
        "style":          "defensive",
    },
    "Hungary": {
        "key_players":    ["Szoboszlai", "Sallai"],
        "scouting_notes": "Szoboszlai (Liverpool) is the engine. Rest of squad considerably weaker.",
        "star_rating":    7.0,
        "star_form":      7.0,
        "squad_depth":    5.5,
        "tactical_score": 6.0,
        "political_boost":5.5,
        "carrying_factor":7.5,   # Very Szoboszlai-dependent
        "style":          "counter-attack",
    },
    "Albania": {
        "key_players":    ["Asllani", "Bajrami"],
        "scouting_notes": "Asllani (Inter Milan) elite. First major tournament — high motivation.",
        "star_rating":    6.0,
        "star_form":      5.5,
        "squad_depth":    5.0,
        "tactical_score": 5.5,
        "political_boost":6.5,   # First WC appearance
        "carrying_factor":6.0,
        "style":          "defensive",
    },

    # ── Group F ───────────────────────────────────────────────────────────────
    "France": {
        "key_players":    ["Mbappe", "Griezmann", "Tchouameni", "Camavinga"],
        "scouting_notes": "Mbappe best player in world 2025-26. Loaded squad across all lines.",
        "star_rating":    9.9,
        "star_form":      9.5,
        "squad_depth":    9.5,
        "tactical_score": 8.5,
        "political_boost":5.0,
        "carrying_factor":7.5,   # Mbappe key but squad is loaded
        "style":          "counter-attack",
    },
    "Belgium": {
        "key_players":    ["De Bruyne", "Lukaku", "Trossard"],
        "scouting_notes": "KDB (35) still world class. Lukaku remains a physical menace. New gen emerging.",
        "star_rating":    8.5,
        "star_form":      7.5,
        "squad_depth":    7.5,
        "tactical_score": 7.5,
        "political_boost":5.0,
        "carrying_factor":7.0,   # KDB dependent
        "style":          "possession",
    },
    "Wales": {
        "key_players":    ["Wilson", "Ampadu"],
        "scouting_notes": "Post-Bale era. Resilient defensively but lack a match-winner.",
        "star_rating":    5.5,
        "star_form":      5.5,
        "squad_depth":    5.5,
        "tactical_score": 6.0,
        "political_boost":5.5,
        "carrying_factor":5.5,
        "style":          "defensive",
    },
    "Slovakia": {
        "key_players":    ["Skriniar", "Lobotka", "Duda"],
        "scouting_notes": "Skriniar elite defender. Lobotka excellent in midfield. Solid compact unit.",
        "star_rating":    7.0,
        "star_form":      6.5,
        "squad_depth":    6.0,
        "tactical_score": 6.5,
        "political_boost":5.0,
        "carrying_factor":6.0,
        "style":          "defensive",
    },

    # ── Group G ───────────────────────────────────────────────────────────────
    "Spain": {
        "key_players":    ["Yamal", "Pedri", "Gavi", "Morata"],
        "scouting_notes": "Yamal generational at 18-19. Spain's most balanced squad in 20 years.",
        "star_rating":    9.3,
        "star_form":      9.0,
        "squad_depth":    9.5,
        "tactical_score": 9.5,   # De la Fuente tactically excellent
        "political_boost":5.5,
        "carrying_factor":5.0,   # Extremely balanced, no single dependency
        "style":          "tiki-taka",
    },
    "Portugal": {
        "key_players":    ["Ronaldo", "Bernardo Silva", "Leao", "Vitinha"],
        "scouting_notes": "Ronaldo 41 at WC, motivational figure. Bernardo/Leao carry the play.",
        "star_rating":    8.5,
        "star_form":      7.5,   # Ronaldo declining; Leao/BSilva compensate
        "squad_depth":    8.5,
        "tactical_score": 7.5,
        "political_boost":5.0,
        "carrying_factor":7.0,   # Ronaldo still demands the ball
        "style":          "attacking",
    },
    "Switzerland": {
        "key_players":    ["Shaqiri", "Xhaka", "Akanji", "Embolo"],
        "scouting_notes": "High-press, high-energy under Murat Yakin. Wide play with aggressive wing-backs. Won 10 corners vs Qatar in WC2026 opener.",
        "star_rating":    7.5,
        "star_form":      7.5,
        "squad_depth":    7.5,
        "tactical_score": 7.5,
        "political_boost":6.0,
        "carrying_factor":6.5,
        "style":          "high-press",   # Yakin's 3-4-3 with aggressive wing-backs
    },
    "Turkey": {
        "key_players":    ["Calhanoglu", "Guler"],
        "scouting_notes": "Guler (Real Madrid) breakout star. Calhanoglu elite at Inter. Wide, physical wingers. WC2026: 8 corners vs Australia.",
        "star_rating":    7.5,
        "star_form":      7.5,
        "squad_depth":    6.5,
        "tactical_score": 6.5,
        "political_boost":6.5,   # Strong nationalistic drive
        "carrying_factor":6.5,
        "style":          "attacking",    # wide aggressive play → corners; WC2026 confirmed high generator
    },
    "Georgia": {
        "key_players":    ["Kvaratskhelia", "Lochoshvili"],
        "scouting_notes": "Kvara (PSG) is world class but entire attack depends on him.",
        "star_rating":    8.0,
        "star_form":      8.5,
        "squad_depth":    5.5,
        "tactical_score": 6.0,
        "political_boost":7.0,   # Cinderella story, massive motivation
        "carrying_factor":8.5,   # Almost entirely Kvara-dependent
        "style":          "counter-attack",
    },

    # ── Group H ───────────────────────────────────────────────────────────────
    "Germany": {
        "key_players":    ["Musiala", "Wirtz", "Havertz", "Kimmich"],
        "scouting_notes": "Wirtz + Musiala generational duo at 22-23. Deep balanced squad.",
        "star_rating":    9.3,
        "star_form":      9.0,
        "squad_depth":    9.0,
        "tactical_score": 8.5,   # Nagelsmann elite tactician
        "political_boost":5.5,
        "carrying_factor":5.5,   # Very balanced
        "style":          "high-press",
    },
    "Netherlands": {
        "key_players":    ["Van Dijk", "Gakpo", "Reijnders", "Dumfries"],
        "scouting_notes": "Van Dijk elite captain. Reijnders excellent. Gakpo dangerous.",
        "star_rating":    8.0,
        "star_form":      7.5,
        "squad_depth":    7.5,
        "tactical_score": 7.0,
        "political_boost":5.0,
        "carrying_factor":5.5,
        "style":          "attacking",
    },
    "Austria": {
        "key_players":    ["Sabitzer", "Arnautovic", "Laimer"],
        "scouting_notes": "Rangnick's high-intensity pressing system is their biggest weapon.",
        "star_rating":    7.0,
        "star_form":      6.5,
        "squad_depth":    6.5,
        "tactical_score": 8.0,   # Rangnick tactically innovative
        "political_boost":5.0,
        "carrying_factor":5.5,
        "style":          "high-press",
    },
    "Romania": {
        "key_players":    ["Ianis Hagi", "Dragusin", "Stanciu"],
        "scouting_notes": "Hagi name carries weight but squad is limited. Dragusin solid at Tottenham.",
        "star_rating":    5.5,
        "star_form":      5.5,
        "squad_depth":    5.5,
        "tactical_score": 5.5,
        "political_boost":5.5,
        "carrying_factor":6.0,
        "style":          "defensive",
    },

    # ── Group I ───────────────────────────────────────────────────────────────
    "Morocco": {
        "key_players":    ["Hakimi", "Ziyech", "En-Nesyri", "Ounahi"],
        "scouting_notes": "2022 semi-final magic. Regragui's defensive system is world class.",
        "star_rating":    8.0,
        "star_form":      7.5,
        "squad_depth":    7.5,
        "tactical_score": 8.5,   # Regragui elite tactician
        "political_boost":7.5,   # Pan-African pride, defending that 2022 legacy
        "carrying_factor":5.5,
        "style":          "defensive",
    },
    "Senegal": {
        "key_players":    ["Mane", "Gueye", "Sarr", "Diatta"],
        "scouting_notes": "Mane still dangerous at 34. Sarr (Chelsea) electric. AFCON champions.",
        "star_rating":    8.0,
        "star_form":      7.0,
        "squad_depth":    7.0,
        "tactical_score": 7.5,
        "political_boost":6.5,
        "carrying_factor":7.0,
        "style":          "counter-attack",
    },
    "Ivory Coast": {
        "key_players":    ["Haller", "Zaha", "Kessie"],
        "scouting_notes": "Haller clinical when fit. Zaha direct and dangerous. Collective quality.",
        "star_rating":    7.0,
        "star_form":      6.5,
        "squad_depth":    6.5,
        "tactical_score": 6.5,
        "political_boost":6.0,
        "carrying_factor":6.0,
        "style":          "attacking",
    },
    "Tunisia": {
        "key_players":    ["Msakni", "Khazri", "Slimane"],
        "scouting_notes": "Disciplined defensive unit. Limited ceiling but hard to beat.",
        "star_rating":    5.5,
        "star_form":      5.5,
        "squad_depth":    5.5,
        "tactical_score": 5.5,
        "political_boost":5.5,
        "carrying_factor":6.0,
        "style":          "defensive",
    },

    # ── Group J ───────────────────────────────────────────────────────────────
    "Egypt": {
        "key_players":    ["Salah", "Trezeguet", "Elneny"],
        "scouting_notes": "Salah (Liverpool) elite — but Egypt is almost entirely his team. Deep risk.",
        "star_rating":    9.0,
        "star_form":      8.5,
        "squad_depth":    5.5,   # Huge drop off after Salah
        "tactical_score": 6.0,
        "political_boost":6.5,
        "carrying_factor":9.5,   # Most Salah-dependent team in the world
        "style":          "counter-attack",
    },
    "South Africa": {
        "key_players":    ["Zungu", "Dolly", "Bafana Bafana"],
        "scouting_notes": "Historic significance. Organised but limited individual quality.",
        "star_rating":    5.0,
        "star_form":      5.0,
        "squad_depth":    5.0,
        "tactical_score": 5.0,
        "political_boost":7.0,   # Historical/political significance
        "carrying_factor":5.5,
        "style":          "defensive",
    },
    "Cameroon": {
        "key_players":    ["Aboubakar", "Anguissa", "Choupo-Moting"],
        "scouting_notes": "Anguissa (Napoli) world-class engine. Aboubakar lethal in big moments.",
        "star_rating":    7.0,
        "star_form":      6.5,
        "squad_depth":    6.5,
        "tactical_score": 6.0,
        "political_boost":6.0,
        "carrying_factor":6.0,
        "style":          "counter-attack",
    },
    "Nigeria": {
        "key_players":    ["Osimhen", "Lookman", "Iheanacho"],
        "scouting_notes": "Osimhen + Lookman = top-5 strike partnership in tournament. Super Eagles hungry.",
        "star_rating":    8.5,
        "star_form":      8.0,
        "squad_depth":    7.5,
        "tactical_score": 6.5,
        "political_boost":6.5,
        "carrying_factor":7.0,
        "style":          "attacking",
    },

    # ── Group K ───────────────────────────────────────────────────────────────
    "Japan": {
        "key_players":    ["Kubo", "Mitoma", "Endo", "Tomiyasu"],
        "scouting_notes": "Most European-based players ever. Tactical discipline + Kubo magic.",
        "star_rating":    7.5,
        "star_form":      7.5,
        "squad_depth":    7.5,
        "tactical_score": 8.0,   # Highly disciplined system
        "political_boost":5.5,
        "carrying_factor":5.0,   # Very collective — no single dependency
        "style":          "high-press",
    },
    "South Korea": {
        "key_players":    ["Son", "Lee Kang-in", "Kim Min-jae"],
        "scouting_notes": "Son (34) still elite, likely last WC. Lee Kang-in and Kim Min-jae excellent.",
        "star_rating":    8.5,
        "star_form":      7.5,
        "squad_depth":    7.0,
        "tactical_score": 7.0,
        "political_boost":5.5,
        "carrying_factor":8.0,   # Very Son-dependent
        "style":          "counter-attack",
    },
    "Australia": {
        "key_players":    ["Leckie", "Hrustic", "Irvine"],
        "scouting_notes": "Socceroos pressing, athletic. WC2026: 5 corners vs Turkey in 2-0 win. Physical wide play generates corners.",
        "star_rating":    6.0,
        "star_form":      5.5,
        "squad_depth":    6.0,
        "tactical_score": 6.5,
        "political_boost":5.5,
        "carrying_factor":6.0,
        "style":          "high-press",   # Arnold's Socceroos: high-energy pressing, wide runners
    },
    "Indonesia": {
        "key_players":    ["Marselino", "Ragnar Oratmangoen", "Jay Idzes"],
        "scouting_notes": "First ever WC. Netherlands-born stars give quality. Emotional overperformance likely.",
        "star_rating":    5.0,
        "star_form":      4.5,
        "squad_depth":    4.5,
        "tactical_score": 5.0,
        "political_boost":8.0,   # First WC ever, 280m population
        "carrying_factor":6.0,
        "style":          "defensive",
    },

    # ── Group L ───────────────────────────────────────────────────────────────
    "Saudi Arabia": {
        "key_players":    ["Al-Dawsari", "Al-Shahrani", "Brozovic"],
        "scouting_notes": "State-backed program, massive resources. WC2026: 33.2% poss vs Uruguay, 4 corners at home — deep defensive block style confirmed.",
        "star_rating":    6.5,
        "star_form":      6.0,
        "squad_depth":    6.0,
        "tactical_score": 6.5,
        "political_boost":7.5,
        "carrying_factor":6.0,
        "style":          "defensive",   # WC2026 confirmed: sits deep, absorbs pressure, low corners
    },
    "Iran": {
        "key_players":    ["Taremi", "Jahanbakhsh", "Beiranvand"],
        "scouting_notes": "Taremi (Inter Milan) world class. Political turbulence reduces team cohesion.",
        "star_rating":    7.0,
        "star_form":      7.0,
        "squad_depth":    6.5,
        "tactical_score": 6.5,
        "political_boost":4.0,   # Internal political tensions, boycotts
        "carrying_factor":7.0,
        "style":          "defensive",
    },
    "Iraq": {
        "key_players":    ["Ameen Al-Dakhil", "Mohanad Ali"],
        "scouting_notes": "First WC in decades. Al-Dakhil (Burnley) solid defender. National pride enormous.",
        "star_rating":    5.5,
        "star_form":      5.5,
        "squad_depth":    5.5,
        "tactical_score": 5.5,
        "political_boost":7.5,   # First WC in 40 years
        "carrying_factor":6.0,
        "style":          "defensive",
    },
    "Uzbekistan": {
        "key_players":    ["Shomurodov", "Khamdamov"],
        "scouting_notes": "Shomurodov (Roma) is their ceiling. First WC — motivated but inexperienced.",
        "star_rating":    5.5,
        "star_form":      5.5,
        "squad_depth":    5.5,
        "tactical_score": 5.5,
        "political_boost":6.5,   # First WC
        "carrying_factor":6.5,
        "style":          "defensive",
    },

    # ── NEW TEAMS IN REAL WC 2026 DRAW ────────────────────────────────────────

    # Group A
    "Czech Republic": {
        "key_players":    ["Patrik Schick", "Tomas Soucek", "Vladimir Coufal"],
        "scouting_notes": "Schick lethal when fit. Soucek engine in midfield. Solid European qualifier.",
        "star_rating":    7.0,
        "star_form":      6.5,
        "squad_depth":    6.5,
        "tactical_score": 7.0,
        "political_boost":5.5,
        "carrying_factor":7.0,   # Schick-dependent in attack
        "style":          "counter-attack",
    },

    # Group B
    "Bosnia and Herzegovina": {
        "key_players":    ["Edin Dzeko", "Miralem Pjanic", "Sehic"],
        "scouting_notes": "Dzeko legend in twilight. Young talent emerging. Motivated first WC since 2014.",
        "star_rating":    6.0,
        "star_form":      5.5,
        "squad_depth":    5.5,
        "tactical_score": 5.5,
        "political_boost":6.5,
        "carrying_factor":6.5,
        "style":          "counter-attack",
    },
    "Qatar": {
        "key_players":    ["Akram Afif", "Almoez Ali", "Saad Al-Sheeb"],
        "scouting_notes": "2022 hosts improved. Afif Golden Ball AFCON. Asian Champions. Technically drilled.",
        "star_rating":    6.0,
        "star_form":      6.5,
        "squad_depth":    6.0,
        "tactical_score": 7.0,   # Well-coached by Lopez
        "political_boost":6.5,
        "carrying_factor":6.5,
        "style":          "possession",
    },

    # Group C
    "Haiti": {
        "key_players":    ["Duckens Nazon", "Frantzdy Pierrot"],
        "scouting_notes": "Historic qualification. Physical and direct. Massive underdog motivation.",
        "star_rating":    4.5,
        "star_form":      4.5,
        "squad_depth":    4.5,
        "tactical_score": 4.5,
        "political_boost":7.5,   # Extraordinary national pride given country's challenges
        "carrying_factor":5.5,
        "style":          "defensive",
    },
    "Scotland": {
        "key_players":    ["Andy Robertson", "Scott McTominay", "Che Adams"],
        "scouting_notes": "McTominay standout at Napoli. Robertson world-class. Best Scotland squad in decades.",
        "star_rating":    7.0,
        "star_form":      7.0,
        "squad_depth":    6.5,
        "tactical_score": 7.0,
        "political_boost":6.5,
        "carrying_factor":6.0,
        "style":          "high-press",
    },

    # Group E
    "Curacao": {
        "key_players":    ["Cuco Martina", "Leandro Bacuna"],
        "scouting_notes": "Caribbean talent pool via Netherlands. Limited top-level experience.",
        "star_rating":    4.5,
        "star_form":      4.5,
        "squad_depth":    4.5,
        "tactical_score": 5.0,
        "political_boost":6.5,   # First ever WC
        "carrying_factor":6.0,
        "style":          "defensive",
    },

    # Group F
    "Sweden": {
        "key_players":    ["Dejan Kulusevski", "Viktor Gyokeres", "Alexander Isak"],
        "scouting_notes": "Gyokeres scoring machine at Sporting CP. Kulusevski Spurs quality. Isak elite at Newcastle.",
        "star_rating":    8.0,
        "star_form":      8.5,   # Gyokeres arguably best striker in Europe 2025
        "squad_depth":    7.5,
        "tactical_score": 7.0,
        "political_boost":5.5,
        "carrying_factor":6.5,
        "style":          "counter-attack",
    },

    # Group G
    "New Zealand": {
        "key_players":    ["Chris Wood", "Clayton Lewis", "Liberato Cacace"],
        "scouting_notes": "Wood experienced Premier League striker. All Whites punching above weight.",
        "star_rating":    5.5,
        "star_form":      5.5,
        "squad_depth":    5.0,
        "tactical_score": 5.5,
        "political_boost":6.0,
        "carrying_factor":7.0,   # Wood dependent
        "style":          "defensive",
    },

    # Group H
    "Cabo Verde": {
        "key_players":    ["Julio Tavares", "Stopira", "Jamiro Monteiro"],
        "scouting_notes": "AFCON semi-finalists. Organised and dangerous on the break. Underdog spirit.",
        "star_rating":    5.5,
        "star_form":      5.5,
        "squad_depth":    5.0,
        "tactical_score": 5.5,
        "political_boost":7.0,   # First WC — massive achievement for small nation
        "carrying_factor":6.0,
        "style":          "counter-attack",
    },

    # Group I
    "Norway": {
        "key_players":    ["Erling Haaland", "Martin Odegaard", "Alexander Sorloth"],
        "scouting_notes": "Haaland best striker on the planet. Odegaard world-class creator. Norway's golden gen.",
        "star_rating":    9.5,   # Haaland is arguably the world's best striker
        "star_form":      9.5,
        "squad_depth":    7.5,
        "tactical_score": 7.5,
        "political_boost":6.0,
        "carrying_factor":8.0,   # Very Haaland-centric
        "style":          "counter-attack",
    },

    # Group J
    "Algeria": {
        "key_players":    ["Riyad Mahrez", "Islam Slimani", "Youcef Atal"],
        "scouting_notes": "Mahrez quality at 35. AFCON champions. Technically strong midfield.",
        "star_rating":    7.5,
        "star_form":      7.0,
        "squad_depth":    6.5,
        "tactical_score": 6.5,
        "political_boost":6.5,
        "carrying_factor":7.0,   # Mahrez dependent
        "style":          "counter-attack",
    },
    "Jordan": {
        "key_players":    ["Yazan Al-Naimat", "Musa Al-Taamari"],
        "scouting_notes": "Surprise qualifiers. Disciplined defensive structure. Al-Taamari danger man.",
        "star_rating":    5.0,
        "star_form":      5.0,
        "squad_depth":    5.0,
        "tactical_score": 5.5,
        "political_boost":7.0,   # First WC ever
        "carrying_factor":6.0,
        "style":          "defensive",
    },

    # Group K
    "Congo DR": {
        "key_players":    ["Cedric Bakambu", "Chancel Mbemba", "Silas Wamangituka"],
        "scouting_notes": "Physical and direct. Bakambu experienced. Strong CAF showing to qualify.",
        "star_rating":    6.0,
        "star_form":      5.5,
        "squad_depth":    5.5,
        "tactical_score": 5.5,
        "political_boost":7.0,   # First WC in decades
        "carrying_factor":6.5,
        "style":          "counter-attack",
    },

    # Group L
    "Ghana": {
        "key_players":    ["Thomas Partey", "Mohammed Kudus", "Antoine Semenyo"],
        "scouting_notes": "Kudus (West Ham) world-class talent. Partey leader. Exciting attacking options.",
        "star_rating":    7.5,
        "star_form":      7.5,
        "squad_depth":    7.0,
        "tactical_score": 6.5,
        "political_boost":6.5,
        "carrying_factor":6.5,
        "style":          "attacking",
    },
    "Croatia": {
        "key_players":    ["Luka Modric", "Mateo Kovacic", "Andrej Kramaric"],
        "scouting_notes": "Modric 40 — last WC, enormous motivation. Kovacic (City) still elite. 2022 bronze medalists.",
        "star_rating":    8.0,
        "star_form":      7.5,
        "squad_depth":    7.5,
        "tactical_score": 8.0,   # Dalic tactically excellent
        "political_boost":6.5,
        "carrying_factor":7.5,   # Modric orchestrates everything
        "style":          "possession",
    },
}


# ── Public API ────────────────────────────────────────────────────────────────

def get_intel_features(team: str) -> dict[str, float]:
    """
    Return the numeric intel feature dict for a team.
    Falls back to conservative defaults for any unknown team.
    """
    d = TEAM_INTEL.get(team, {})
    return {
        "star_rating":     d.get("star_rating",     5.5),
        "star_form":       d.get("star_form",        5.5),
        "squad_depth":     d.get("squad_depth",      5.0),
        "tactical_score":  d.get("tactical_score",   5.0),
        "political_boost": d.get("political_boost",  5.0),
        "carrying_factor": d.get("carrying_factor",  5.5),
        "style_score":     STYLE_ENCODING.get(d.get("style", "defensive"), 0.5),
    }


def build_intel_df() -> pd.DataFrame:
    """Return a DataFrame with one row per team for the full scouting report."""
    rows = []
    for team, d in TEAM_INTEL.items():
        row = {
            "team":        team,
            "key_players": ", ".join(d.get("key_players", [])),
            "notes":       d.get("scouting_notes", ""),
        }
        row.update(get_intel_features(team))
        rows.append(row)
    return (pd.DataFrame(rows)
              .sort_values("star_rating", ascending=False)
              .reset_index(drop=True))


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    df = build_intel_df()
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 140)
    pd.set_option("display.max_colwidth", 45)

    print("\n" + "="*140)
    print("  WC 2026 — SCOUTING & INTELLIGENCE REPORT")
    print("="*140)
    print(df[[
        "team", "key_players",
        "star_rating", "star_form", "squad_depth",
        "tactical_score", "political_boost", "carrying_factor",
    ]].to_string(index=True))
    print("="*140)
    print("\ncarrying_factor: higher = more dependent on one star player (riskier)")
    print("political_boost: host nations / first WC / national-pride factor\n")

    print("\n--- TOP 10 BY STAR RATING ---")
    for _, r in df.head(10).iterrows():
        print(f"  {r['team']:<22} star={r['star_rating']:.1f}  form={r['star_form']:.1f}  "
              f"depth={r['squad_depth']:.1f}  tactics={r['tactical_score']:.1f}  "
              f"carry_risk={r['carrying_factor']:.1f}")

    print("\n--- BIGGEST POLITICAL BOOSTS ---")
    for _, r in df.sort_values("political_boost", ascending=False).head(8).iterrows():
        print(f"  {r['team']:<22} political_boost={r['political_boost']:.1f}  "
              f"  notes: {r['notes'][:60]}")

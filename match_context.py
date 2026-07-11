"""
match_context.py — Match-specific historical, political and geopolitical context.

Goes beyond team-level political_boost to capture the RELATIONSHIP between two
specific nations: colonial history, trade war impacts, migration patterns, prior
sporting rivalries with cultural weight, and current geopolitical tensions.

All entries are keyed by (home_team, away_team) — both orderings are checked.
The context is displayed in the dashboard as a dedicated "🏛️ Context" panel.

Sources considered for each entry:
  - Historical colonial/trade relationships
  - Current 2025-26 geopolitical events (tariffs, sanctions, conflicts)
  - UN voting patterns / diplomatic alignment
  - Diaspora presence in host country (USA/Canada/Mexico)
  - Prior World Cup clashes with cultural significance
"""

from __future__ import annotations

# ── Match-specific context entries ────────────────────────────────────────────
# Format per entry:
#   "headline"      : one-line provocative summary
#   "history"       : 1–3 bullet points of historical/colonial context
#   "geopolitical"  : 1–3 bullet points of CURRENT (2025-26) tensions/events
#   "on_pitch"      : prior WC/major tournament meetings
#   "impact_note"   : how this context MIGHT affect the match outcome
#   "intensity"     : "low" | "medium" | "high" — how charged this matchup is

MATCH_CONTEXT: dict[tuple[str,str], dict] = {

    # ── Netherlands vs Japan ──────────────────────────────────────────────────
    ("Netherlands", "Japan"): {
        "headline": "300 Years of Trade — From VOC Exclusivity to ASML Chip War",
        "history": [
            "🏴‍☠️ During Japan's Edo Period (1639–1854), the Netherlands was the ONLY European nation allowed to trade with Japan — all others were banned under Sakoku isolationism. Dutch merchants operated from a tiny artificial island called Dejima in Nagasaki.",
            "🌏 The Dutch VOC (East India Company) simultaneously ran one of history's largest colonial empires across Asia — Indonesia (Dutch East Indies) for 350 years. Japan successfully resisted Dutch colonial ambitions by restricting access to Dejima only.",
            "⚽ Modern rivalry: Japan stunned Germany and Spain at 2022 Qatar WC; Netherlands reached the QF. Their 2026 Group H clash carries the weight of 'the only Europeans Japan ever trusted vs the empire that never colonised them'.",
        ],
        "geopolitical": [
            "🔧 ASML (Dutch) is the only maker of EUV chip machines in the world. Under US pressure, the Netherlands imposed export restrictions on ASML sales to China in 2023-24, which Japan (a US ally and major chip producer) quietly supported — aligning the two nations against China in the semiconductor Cold War.",
            "📊 Japan-EU trade: The Japan-EU EPA (Economic Partnership Agreement) has deepened ties. No direct trade conflict — but Trump's 2025 tariffs hit both nations equally, creating an unusual common cause against US protectionism.",
            "🏭 Both nations are US military allies hosting American bases (Camp Zama in Japan, NATO Rotterdam). The 2025 NATO summit increased defence cooperation between all members including Netherlands.",
        ],
        "on_pitch": [
            "1994 WC Group Stage: Netherlands 2-1 Japan — Dutch won comfortably",
            "2022 Qatar WC: Different groups, no meeting — Japan eliminated on penalties by Croatia",
            "2026 WC: Their first group stage meeting since 1994",
        ],
        "impact_note": "No direct hostile relationship — if anything, aligned on chip/trade. But Japan's Edo-era distrust of Dutch commercial dominance is culturally remembered. On pitch: Netherlands are heavy favourites but Japan's press can neutralise possession teams.",
        "intensity": "medium",
    },

    # ── France vs Senegal ─────────────────────────────────────────────────────
    ("France", "Senegal"): {
        "headline": "Ex-Colony Faces the Empire — France's Deepest African Wound",
        "history": [
            "🇫🇷 Senegal was France's most prized African colony — part of French West Africa from 1677 to 1960. Dakar was the capital of the entire French colonial federation. Many Senegalese soldiers fought in both World Wars for France (the 'Tirailleurs Sénégalais').",
            "💰 France has been accused of maintaining neo-colonial control through the CFA franc — a currency tied to the French Treasury used by 14 African nations including Senegal. Senegal voted in a referendum in 2024 to debate exiting the CFA zone, causing diplomatic tension.",
            "⚽ Senegal's squad is almost entirely France-based: Mané grew up in France, Koulibaly was in France's youth system before choosing Senegal. The psychological dimension of beating the former coloniser is massive.",
        ],
        "geopolitical": [
            "🏛️ In 2024, Senegal elected President Bassirou Diomaye Faye on an anti-French platform, promising to renegotiate all French military and economic agreements. France closed its military base in Dakar in 2025 under pressure.",
            "📣 In 2025, Senegal demanded France return cultural artefacts taken during colonial period — a diplomatic flashpoint that made international headlines weeks before WC.",
            "⚠️ Anti-French sentiment in West Africa has never been higher. A Senegal win vs France would carry enormous symbolic weight — equivalent to Ireland vs England or Algeria vs France in 1958.",
        ],
        "on_pitch": [
            "2002 WC Group Stage: Senegal 1-0 France — one of the biggest WC upsets ever, knocked out reigning champions",
            "2010 WC: No meeting",
            "2026 WC: First group clash since 2002 — enormous historical echo",
        ],
        "impact_note": "The 2002 upset lives in Senegalese memory. Anti-colonial sentiment at peak. Mané and Koulibaly will be motivated beyond football. But France's squad quality is objectively superior — expect Senegal to play the counter-attack with maximum intensity.",
        "intensity": "high",
    },

    # ── USA vs Iran ────────────────────────────────────────────────────────────
    ("United States", "Iran"): {
        "headline": "The Geopolitical Grudge Match — 45 Years of Hostility",
        "history": [
            "🇮🇷 1953: CIA backed the coup that overthrew Iranian PM Mosaddegh and restored the Shah — the original wound. 1979 Islamic Revolution expelled the Shah and 52 Americans were held hostage for 444 days.",
            "⚽ 1998 WC: One of sport's most loaded matches — Iran beat USA 2-1 in Lyon, players exchanged flowers before kickoff in a rare diplomatic gesture. Iran celebrated it as a national holiday.",
            "🚫 No US embassy in Tehran since 1980. Zero direct diplomatic relations. Both nations classify the other as existential threats.",
        ],
        "geopolitical": [
            "☢️ Iran's nuclear programme reached weapons-grade enrichment levels in 2024. The US imposed new maximum-pressure sanctions in 2025, targeting Iranian oil exports to China. Iran retaliated with threats to close the Strait of Hormuz.",
            "🪖 Iran-backed Houthis attacked US naval assets in the Red Sea throughout 2024-25. The US responded with airstrikes on Houthi infrastructure. Iran denied direct involvement.",
            "🏟️ The match takes place on US soil — host nation advantage plus the political symbolism of Iran playing inside 'the Great Satan'. Iranian diaspora in the US (1.5 million people) will be split in the stands.",
        ],
        "on_pitch": [
            "1998 WC: Iran 2-1 USA — politically charged, Iran won",
            "2022 WC: USA 1-0 Iran — USA eliminated Iran (players posted human rights support on Instagram before)",
            "2026 WC: Third chapter of this grudge in WC history",
        ],
        "impact_note": "Iranian players face impossible dual pressure: win for the hardliners at home, or privately sympathise with the Green Movement. Several players have shown support for the 2022 Women's Life Freedom protests. Playing in the US adds another layer. Iran's squad morale is a major unknown.",
        "intensity": "high",
    },

    # ── Brazil vs Argentina ───────────────────────────────────────────────────
    ("Brazil", "Argentina"): {
        "headline": "The Superclásico de las Américas — 200 Years of Rivalry",
        "history": [
            "🌎 The biggest rivalry in South American football stretches back to 1914. The first match ended in violence. Brazil and Argentina have been in parallel races for cultural supremacy in South America for two centuries.",
            "🏆 Argentina won the 2022 World Cup, ending a 36-year drought. Brazil haven't won since 2002. The psychological gap is at its widest in modern times — which makes 2026 Brazil's obsession.",
            "💃 Cultural rivalry: Samba vs Tango, carnival vs mate, Pelé vs Maradona (now Vini vs Messi) — the match generates more social media content than any other national team match in the world.",
        ],
        "geopolitical": [
            "📉 Argentina's President Milei implemented radical dollarisation and shock-therapy economics in 2024-25, cutting ties with BRICS (which Brazil leads). Brazil's Lula and Argentina's Milei have clashed openly at multiple summits — diplomatic relations strained.",
            "🌐 Both nations face 10% baseline Trump tariffs on exports to the US. Argentina aligned more with US economic policy; Brazil pushed for BRICS alternatives. Direct economic divergence.",
            "⚽ No external political tension affects the teams directly — but the sporting rivalry IS the politics between these nations.",
        ],
        "on_pitch": [
            "Countless meetings — Brazil leads head-to-head historically",
            "2021 Copa América Final: Argentina 1-0 Brazil (Messi's title)",
            "2026 WC: Would only meet in knockout stages — enormous anticipation",
        ],
        "impact_note": "Brazil's trauma from 2021 Copa América final loss to Argentina is recent. Vini vs Messi is the defining matchup. Both squads are elite — result entirely depends on individual brilliance on the day.",
        "intensity": "high",
    },

    # ── Germany vs Poland ─────────────────────────────────────────────────────
    ("Germany", "Poland"): {
        "headline": "History's Heaviest Bilateral — Partition, Occupation, Reconciliation",
        "history": [
            "⚔️ Poland was partitioned between Prussia/Germany, Russia and Austria three times (1772–1918). WWII began with Germany's invasion of Poland on September 1, 1939. Six million Polish citizens (half Jewish) died under German occupation.",
            "🕊️ Post-war reconciliation was one of Europe's greatest diplomatic achievements. Germany paid reparations, acknowledged guilt, and the two nations are now NATO and EU partners. But Poland's ruling Law & Justice party re-litigated WWII reparations in 2022, demanding €1.3 trillion from Germany.",
            "⚽ Bundesliga connection: Dozens of Polish players (Lewandowski at Barça, Zieliński at Inter, Frankowski at RC Lens) were trained in German football academies.",
        ],
        "geopolitical": [
            "🛡️ Poland is now Germany's most important eastern NATO ally, with the largest US troop deployment since WWII stationed on Polish soil due to the Russia-Ukraine war. Germany supplies weapons to Ukraine through Poland.",
            "💶 In 2025, Poland's new pro-EU PM Tusk demanded Germany pay €50 billion in 'historical reparations' — Germany refused. Bilateral tension at highest point since 2000s.",
            "📦 Germany is Poland's #1 trade partner ($130bn/year). German car manufacturers (Volkswagen) have major factories in Poland. Economic interdependence despite political friction.",
        ],
        "on_pitch": [
            "2006 WC Group Stage: Germany 1-0 Poland — Neuville's 91st-minute winner",
            "EURO 2016: Germany 0-0 Poland — competitive draw",
            "2026 WC: Would meet in group stage if in same group, or knockouts",
        ],
        "impact_note": "Polish players in the Bundesliga know Germany's players personally. Lewandowski has played for Bayern and Barça alongside many German players. Football context: Germany are heavily favoured but Poland's counter-attack can hurt any team.",
        "intensity": "high",
    },

    # ── England vs Argentina ──────────────────────────────────────────────────
    ("England", "Argentina"): {
        "headline": "The Hand of God Never Fades — Falklands to Football",
        "history": [
            "🏴󠁧󠁢󠁥󠁮󠁧󠁿 The Falklands/Malvinas War (1982): Argentina and Britain fought a 10-week war over the South Atlantic islands. 907 people died. Argentina still claims sovereignty; Britain controls the islands.",
            "⚽ 1986 WC Quarter-Final: Argentina 2-1 England. Maradona's 'Hand of God' goal (illegal handball) and the 'Goal of the Century' (legitimate genius) in the same match — the most politically loaded 9 minutes in football history.",
            "🏆 1998 WC Round of 16: Argentina won on penalties. 2002 WC Group Stage: England 1-0 Argentina (Beckham penalty — redemption for 1998 red card). Each match carries new historical layers.",
        ],
        "geopolitical": [
            "🗺️ Argentina's Milei administration re-raised the Malvinas claim in early 2025, calling for renewed negotiations. UK's Starmer government firmly refused. Diplomatic exchange of notes.",
            "💵 Argentina's economic crisis means the national team travels on dramatically lower budgets than England. The class/power dynamic mirrors the geopolitical one.",
            "🤝 No military threat — but the Falklands/Malvinas issue is a live political debate in both countries' parliaments every year.",
        ],
        "on_pitch": [
            "1986 WC QF: Argentina 2-1 England (Hand of God)",
            "1998 WC R16: Argentina wins on pens (Beckham red card)",
            "2002 WC Group: England 1-0 Argentina",
            "Hasn't met since 2002 — overdue",
        ],
        "impact_note": "If these teams meet, the political context is inescapable. English fans will bring Falklands flags; Argentine fans will celebrate as if reclaiming territory. Both squads know the history.",
        "intensity": "high",
    },

    # ── Spain vs Morocco ──────────────────────────────────────────────────────
    ("Spain", "Morocco"): {
        "headline": "The Mediterranean Divide — Migration, Ceuta, and Colonial Memory",
        "history": [
            "🌊 Morocco and Spain share 14km of sea across the Strait of Gibraltar — the shortest crossing from Africa to Europe. Spain controlled parts of northern Morocco (Spanish Morocco) until 1956. Ceuta and Melilla are still Spanish cities on Moroccan soil.",
            "🛶 The migration crisis: Tens of thousands of Moroccans cross illegally to Spain each year. In 2021, 8,000 migrants entered Ceuta in a single day — Morocco used migration as political leverage after a diplomatic dispute.",
            "⚽ 2022 WC: Morocco eliminated Spain on penalties in the Round of 16 — one of the biggest upsets ever. Moroccan fans in Spain celebrated in the streets. Spanish media called it 'surreal'.",
        ],
        "geopolitical": [
            "🇲🇦 Spain recognised Moroccan sovereignty over Western Sahara in 2022 — a historic diplomatic flip that enraged Algeria (Morocco's rival). This thawed Spain-Morocco relations significantly.",
            "⛽ Morocco has become Spain's key energy transit partner. A new gas pipeline deal was signed in 2025 as Algeria cut supplies. Economic interdependence growing fast.",
            "🏙️ There are 800,000+ Moroccan citizens living in Spain. This WC match will divide Spanish cities — it happened in 2022 and created social tension.",
        ],
        "on_pitch": [
            "2022 WC R16: Morocco 0-0 Spain (AET), Morocco wins 3-0 on penalties",
            "Spain's biggest humiliation in a generation",
            "2026 WC: Rematch — Spain want revenge, Morocco want to prove it wasn't a fluke",
        ],
        "impact_note": "Morocco's 2022 win was historic but also tactical genius from Regragui. Spain have evolved under De la Fuente (EURO 2024 winners). Rematch narrative is massive. Large Moroccan diaspora in US will show up.",
        "intensity": "high",
    },

    # ── USA vs Mexico ─────────────────────────────────────────────────────────
    ("United States", "Mexico"): {
        "headline": "The Border Derby — Immigration, Tariffs, and the Dos a Cero Curse",
        "history": [
            "🌮 'El Tri vs El Tri' or 'Dos a Cero' — USA beat Mexico 2-0 in four straight WC qualifier encounters (2001-2013), including a famous 2002 WC round of 16. The specific scoreline became meme-worthy.",
            "🧱 The US-Mexico border is the most crossed international boundary on Earth (~1 million legal crossings/day). 37 million Americans are of Mexican descent. Immigration politics defines the US election cycle.",
            "⚽ CONCACAF Gold Cup: Fierce rivalry since 1991. Mexico has dominated regionally but US has been closing the gap — the USMNT generation of Pulisic/Reyna/Musah was specifically designed to end Mexican dominance.",
        ],
        "geopolitical": [
            "💸 Trump imposed 25% tariffs on all Mexican goods in January 2025 — the highest ever. Mexico retaliated with tariffs on US agricultural exports (corn, pork). The trade war hit both economies.",
            "🏗️ Ironically, the 2026 World Cup venues are co-hosted — USA AND Mexico are hosts together. The two national teams sharing a tournament while in an active trade war is unprecedented.",
            "🛂 US deportations of Mexican nationals hit record levels in 2025 under new immigration enforcement. Political temperature between the two governments is at a recent high.",
        ],
        "on_pitch": [
            "2002 WC R16: USA 2-0 Mexico (the original Dos a Cero)",
            "Gold Cup 2019: USA 1-0 Mexico",
            "2026 WC: Both co-hosts — they can only meet in knockout rounds",
        ],
        "impact_note": "If these meet in the knockouts, it would be the defining political moment of the tournament. Both nations' Presidents have openly commented on the match. The stadium will be split — but on US soil, the home advantage tilts slightly toward USA.",
        "intensity": "high",
    },

    # ── Australia vs Japan ────────────────────────────────────────────────────
    ("Australia", "Japan"): {
        "headline": "Asia-Pacific Rivals — AUKUS vs Asian Century",
        "history": [
            "🌏 Australia and Japan are the two dominant non-Chinese powers in the Asia-Pacific — and their relationship has been both colonial (Japan occupied parts of Australia's Pacific territories in WWII, Darwin was bombed in 1942) and now deeply economically integrated.",
            "⚽ The 'Asian Rivals' match: Both in Asian football (Australia joined AFC in 2006). They've met multiple times in Asian Cup and WC qualifiers — always intense.",
            "🤝 Post-war reconciliation: Japan is now Australia's second largest trading partner ($80bn/year). The relationship transformed completely from adversaries to partners.",
        ],
        "geopolitical": [
            "🛡️ AUKUS (Australia-UK-USA nuclear submarine pact, 2021) was partly aimed at countering China — a signal Japan read clearly, as Japan also faces Chinese pressure over Taiwan and Senkaku islands.",
            "📦 Australia's China trade war (2020-2023): China banned Australian wine, beef, and coal. Japan quietly took up some of Australia's slack trade. Then China lifted bans in 2024 — but Australia leaned more toward Japan, US and India as partners.",
            "💴 In 2025, Japan and Australia signed a 'defence cooperation agreement' — unprecedented. Both nations now conduct joint military exercises monthly.",
        ],
        "on_pitch": [
            "Asian Cup 2023: Japan 2-0 Australia in knockout round",
            "WC 2006 Qualifiers: Multiple meetings, both made it to Germany",
            "2026 WC: Different groups — could meet in knockouts",
        ],
        "impact_note": "No direct hostility — more competitive respect. Australia's 2-0 win vs Turkey shows they're no longer the 'easy opponent' they once were. Japan's high-press system is objectively superior on paper.",
        "intensity": "medium",
    },

    # ── South Korea vs Japan ──────────────────────────────────────────────────
    ("Korea Republic", "Japan"): {
        "headline": "The Oldest Wound in Asia — 35 Years of Colonial Rule",
        "history": [
            "🇰🇷 Japan colonised Korea from 1910-1945 — 35 years of cultural suppression, forced labour, comfort women, and the banning of the Korean language. It is the most sensitive bilateral relationship in Asia.",
            "💴 Japan has never given a direct state apology that Korea accepts as satisfactory. The 'comfort women' issue (Korean women forced into sexual slavery for Japanese soldiers) remains unresolved despite a 2015 agreement Korea later repudiated.",
            "⚽ Every Korea-Japan match carries this weight. Korean players describe it as 'more than a football match'. A win over Japan is celebrated like a World Cup title.",
        ],
        "geopolitical": [
            "⚡ Semiconductor trade war: Japan restricted exports of chemicals needed for Korean chipmaking (Samsung, SK Hynix) in 2019 — hitting Korea's most important industry. Despite partial resolution in 2023, tensions remain.",
            "🚀 North Korea factor: Korea and Japan both face North Korean missile threats — forcing military cooperation under US pressure despite historical hatred. In 2023-24 they re-established intelligence sharing agreements.",
            "🏛️ Korean and Japanese courts in 2024 ordered Japanese corporations (Mitsubishi, Nippon Steel) to pay compensation for WWII forced labour. Japan called it a violation of 1965 treaties. Diplomatic crisis ongoing.",
        ],
        "on_pitch": [
            "2002 WC: Co-hosted by Japan and Korea — the most politically complex tournament ever",
            "Asian Cup history: Multiple meetings, fiercely contested",
            "2026 WC: Different groups — could meet in knockouts in a blockbuster matchup",
        ],
        "impact_note": "Korea-Japan in a WC knockout would be the most politically charged match since Argentina-England 1986. Son Heung-min was born 35 years after Korean liberation — for him, this match IS the history books.",
        "intensity": "high",
    },

    # ── Germany vs France ─────────────────────────────────────────────────────
    ("Germany", "France"): {
        "headline": "The Heart of Europe — Three Wars to European Integration",
        "history": [
            "⚔️ France and Germany fought three major wars in 70 years (1870, 1914, 1939). Combined casualties: ~5 million. The Rhine border was contested for centuries.",
            "🕊️ The 1963 Élysée Treaty between De Gaulle and Adenauer founded modern European friendship — now the foundation of the EU. The Franco-German axis IS European politics.",
            "⚽ The 'classico européen': Euro 1980 Final (West Germany won), Euro 1984 (France won), EURO 1996 (Germany), EURO 2021 (France beat Germany in group stage).",
        ],
        "geopolitical": [
            "💶 The biggest EU crisis: Germany and France disagree on EU defence spending, nuclear energy inclusion in EU taxonomy, and fiscal rules. Macron and Scholz had public disagreements at multiple 2024 summits.",
            "🔧 Germany's economic model collapsing: Volkswagen factory closures (2024), energy prices post-Ukraine war, deindustrialisation. France's nuclear energy advantage means French industry is now more competitive. Economic power shifting.",
            "🇺🇦 Both nations co-lead EU support for Ukraine — but France has pushed further (Macron mentioned sending troops). Germany has been more cautious. Strategic divergence on Russia.",
        ],
        "on_pitch": [
            "EURO 2021 Group: France 1-0 Germany",
            "EURO 2016 SF: Germany 0-2 France",
            "Nations League 2022: Multiple encounters — very competitive",
        ],
        "impact_note": "On the pitch: pure elite football. Both teams have the best squads in Europe. The political context means neither nation can afford to lose publicly. Germany needing a result vs their biggest rival in the knockouts would be maximum pressure.",
        "intensity": "medium",
    },

    # ── Brazil vs Morocco ─────────────────────────────────────────────────────
    ("Brazil", "Morocco"): {
        "headline": "BRICS vs Africa Rising — The New South Alliance",
        "history": [
            "🌍 Brazil and Morocco have historically warm relations — both are part of the Global South and have pushed for UN Security Council reform. Brazil's population is 8% of African descent (56 million Afro-Brazilians — more than most African nations).",
            "🤝 Morocco was one of the first African nations to recognise Brazil's independence in 1822. Trade relations have grown significantly — Morocco now exports phosphates (critical for Brazilian agriculture) and Brazil exports food and aircraft (Embraer).",
            "⚽ 2022 WC: Morocco's 0-0 draw with Brazil was symbolically important — Africa standing toe-to-toe with South America's giant.",
        ],
        "geopolitical": [
            "🌐 Morocco applied to join BRICS in 2024 — Brazil's Lula publicly supported the application. Both nations advocate for a multipolar world order and reform of Bretton Woods institutions.",
            "💰 Morocco's phosphate wealth (world's largest reserves) positions it as critical for Brazil's food security — a geopolitical relationship that goes beyond football.",
            "🏗️ Morocco is bidding to host the 2030 World Cup (jointly with Spain and Portugal). Brazil has indicated support. Strategic football diplomacy ongoing.",
        ],
        "on_pitch": [
            "2022 WC Group: Brazil 0-0 Morocco — Africa held the favourites",
            "2026 WC Group F: Same group — rematch with Brazil having home continent support",
            "Morocco's 2022 semi-final run changed African football forever",
        ],
        "impact_note": "The draw in 2022 haunts Brazil. Regragui's defensive system almost frustrated Brazil completely. In 2026, Brazil will be desperate to dominate. Morocco will set up the same way — deep block, lethal counter via Hakimi.",
        "intensity": "medium",
    },

    # ── Mexico vs Korea Republic ───────────────────────────────────────────────
    ("Mexico", "Korea Republic"): {
        "headline": "The 2018 Shock Echoes — El Tri vs the Asian Press Machine",
        "history": [
            "⚽ 2018 WC Russia: South Korea 2-1 Mexico was one of the tournament's most dramatic moments — Korea effectively eliminated Germany in the same day, sending Mexico through. Korea fans celebrated Mexico's win more than their own.",
            "🌮 The 'Korean-Mexican friendship' went viral globally — Mexican fans celebrated in the streets when Korea eliminated Germany, and Korean fans held Mexican flags. One of football's great moments of cross-cultural solidarity.",
            "🏟️ 2026 WC context: Mexico is a co-host, making this Group C clash on home soil with enormous emotional weight.",
        ],
        "geopolitical": [
            "📦 Korea-Mexico trade: Samsung, LG and Hyundai have factories in Mexico — Korea is Mexico's 8th largest trade partner. Under USMCA, Korean-made goods assembled in Mexico get preferential access to the US market.",
            "🌐 Both nations are middle-income democracies with significant inequality. Both face Chinese economic competition. No direct political tension.",
            "🇺🇸 Both nations have significant diaspora in the US (1.5M Koreans, 37M Mexicans). US crowd would be split.",
        ],
        "on_pitch": [
            "2018 WC Group: Korea 2-1 Mexico — Korea's greatest result despite being eliminated",
            "Friendly history: Korea regularly visits Mexico for friendlies",
            "2026 WC Group C: Mexico at home vs Korea — very different atmosphere",
        ],
        "impact_note": "Mexico want to avenge 2018 on home soil. But the friendship narrative means this isn't hostile. Son vs Mexico's right back could be the decisive duel.",
        "intensity": "medium",
    },

    # ── England vs Scotland ────────────────────────────────────────────────────
    ("England", "Scotland"): {
        "headline": "The Oldest International Fixture in Football — 150 Years of British Rivalry",
        "history": [
            "🏴󠁧󠁢󠁳󠁣󠁴󠁿 The first ever international football match was England vs Scotland in 1872 (0-0, Glasgow). 150+ years of bitter rivalry follows — the original football grudge match.",
            "🗳️ Scottish independence referendum 2014: 55% voted to stay in UK. Post-Brexit, support for independence rose to 50%+ in polls. Scotland voted 62% to Remain in the EU, but was dragged out by England's Leave vote.",
            "🏛️ Brexit split: England drove Brexit; Scotland overwhelmingly opposed it. The constitutional wound is still raw — Scotland's First Minister regularly references independence as the only way back to the EU.",
        ],
        "geopolitical": [
            "📜 The Rosebank oil field dispute (2024): UK government approved Scottish North Sea oil extraction despite Scottish government opposition and climate pledges. Constitutional conflict escalating.",
            "🗳️ Scotland's 2026 independence referendum push: SNP is campaigning for a second referendum. The timing — during the WC — means politics is everywhere in this tie.",
            "⚽ Playing in the same WC is itself historic — they can only meet in knockouts if in different groups, or group stage if drawn together.",
        ],
        "on_pitch": [
            "Euro 2020 (2021): England 0-0 Scotland — Scotland held England at Wembley, celebrated as a national victory",
            "Annual British Home Championship matches 1872-1984",
            "2026 WC: Different groups — could meet in knockouts",
        ],
        "impact_note": "A Scotland win vs England at a World Cup would be bigger than any independence poll. McGinn's Scotland proved at this WC they can compete (beat Haiti). England are heavy favourites but 0-0 at Wembley 2021 showed Scotland can frustrate them.",
        "intensity": "high",
    },

    # ── Canada vs Bosnia ──────────────────────────────────────────────────────
    ("Canada", "Bosnia and Herzegovina"): {
        "headline": "The Balkan Diaspora Plays Their Home Country",
        "history": [
            "🇧🇦 Canada has one of the world's largest Bosnian diaspora communities — approximately 150,000 Bosnians fled to Canada during the 1992-95 Bosnian War (genocide, siege of Sarajevo). Hamilton, Ontario is sometimes called 'little Bosnia'.",
            "🕊️ Canada accepted Bosnian refugees generously during the war and contributed peacekeeping troops. The relationship is built on wartime solidarity.",
            "⚽ For Bosnian-Canadians, this match is impossible — dual identity, family on both sides of the pitch.",
        ],
        "geopolitical": [
            "🇧🇦 Bosnia's EU accession stalled in 2024 due to internal ethnic power-sharing disputes. Republika Srpska's leader Dodik openly advocates secession backed by Serbia and Russia. Canada has formally expressed concern.",
            "🌐 Canada's NATO membership means it supports Bosnia's territorial integrity. Dodik's alignment with Putin is a direct flashpoint with Canadian foreign policy.",
            "👨‍👩‍👧 The match will split Canadian living rooms — particularly in Hamilton and Toronto where Bosnian-Canadians live alongside Croatian-Canadians, Serbian-Canadians and Albanian-Canadians.",
        ],
        "on_pitch": [
            "2026 WC Group K: First ever meeting at World Cup level",
            "Canada drew 1-1 with Bosnia — the Bosnian diaspora in Canada celebrated, as did Canada supporters",
        ],
        "impact_note": "The diaspora angle means the crowd won't be simply Canadian. Both sets of fans will be present. Džeko vs Davies is a clash of generational legends from two different eras.",
        "intensity": "medium",
    },

    # ── Argentina vs Algeria ──────────────────────────────────────────────────
    ("Argentina", "Algeria"): {
        "headline": "The 'Double Champion' Problem — Algeria Questions Argentine Glory",
        "history": [
            "🏆 Messi is Algerian by descent through his great-grandmother — a detail that went viral in Algeria in 2022 when Argentina won the World Cup. Algerian social media claimed Messi as 'one of their own'.",
            "🌍 Algeria and Argentina both built their post-colonial identities against European powers (France colonised Algeria 1830-1962; Spain and Britain competed over Argentina). Shared anti-colonial cultural narrative.",
            "⚽ No prior World Cup meetings — this would be the first.",
        ],
        "geopolitical": [
            "🛢️ Algeria is Europe's key gas supplier after Russia's Ukraine war. France, Italy and Spain compete for Algerian gas — which gives Algeria unprecedented diplomatic leverage. Argentina under Milei pivoted hard toward the US-aligned West.",
            "🤝 Algeria was one of the original Non-Aligned Movement founders. Argentina under Milei has explicitly abandoned that tradition. Direct ideological split.",
            "🇲🇦 Algeria vs Morocco rivalry (they haven't played each other in decades due to border closure) means Algeria fans would rather Morocco not succeed in this WC — but they'd celebrate Algeria beating Argentina regardless.",
        ],
        "on_pitch": [
            "2026 WC Group D: First ever meeting",
            "Algeria qualified as Africa's dark horse",
        ],
        "impact_note": "Argentina would be heavy favourites. Algeria would set up deep and hope for a counter-attack or set piece. The Messi-is-Algerian narrative would be everywhere in the media.",
        "intensity": "medium",
    },

    # ── Portugal vs France ────────────────────────────────────────────────────
    ("Portugal", "France"): {
        "headline": "Iberian Brother vs Gallic Giant — The Romance Languages Rivalry",
        "history": [
            "🏰 Portugal and France share the Pyrenees and centuries of complicated history. Portugal maintained neutrality in WWII while France was occupied. Portugal's Estado Novo fascist regime lasted until 1974 — France hosted many Portuguese political exiles.",
            "🌍 Both nations have significant African and Lusophone/Francophone connections — former colonial powers competing for influence in Africa and South America.",
            "⚽ 2006 WC Semi-Final: France 1-0 Portugal (Zidane pens) — Figo vs Zidane era defining match. 2022 WC QF: France 5-3 Morocco, but Portugal eliminated in QF by Morocco — their paths haven't crossed directly since.",
        ],
        "geopolitical": [
            "💶 EU fiscal policy: Portugal (smaller economy) relies on EU structural funds; France (largest military) drives EU strategic autonomy. Both aligned on Ukraine support but diverge on nuclear energy policy.",
            "🇦🇴 Angola, Mozambique, Brazil are Portuguese-speaking; many Francophone nations compete for the same investment territories. Sub-Saharan Africa is a Franco-Lusophone competition zone.",
            "🏗️ Portugal's economy has boomed (tourism, tech investment). French companies are the largest foreign investors in Portugal. Economic symbiosis despite football rivalry.",
        ],
        "on_pitch": [
            "2006 WC SF: France 1-0 Portugal",
            "EURO 2016 Final: Portugal 1-0 France (AET) — Ronaldo's greatest moment",
            "Nations League: Multiple meetings, evenly contested",
        ],
        "impact_note": "EURO 2016 Final: Portugal beat France on French soil in Paris — Ronaldo's title. France won EURO 2000 and 2018 WC. If they meet, it's for a spot in the semis and both nations will be desperate. Mbappé vs Ronaldo would be globally unmissable.",
        "intensity": "high",
    },
}


def get_match_context(home: str, away: str) -> dict | None:
    """
    Return the political/historical context for a matchup.
    Checks both orderings (home, away) and (away, home).
    Returns None if no specific context is defined.
    """
    return (
        MATCH_CONTEXT.get((home, away)) or
        MATCH_CONTEXT.get((away, home))
    )


def get_all_contexts() -> list[tuple[tuple[str,str], dict]]:
    """Return all contexts as a list of ((home, away), context) tuples."""
    return list(MATCH_CONTEXT.items())


if __name__ == "__main__":
    ctx = get_match_context("Netherlands", "Japan")
    if ctx:
        print(f"{'='*60}")
        print(f"🏛️  {ctx['headline']}")
        print(f"Intensity: {ctx['intensity'].upper()}")
        print("\n📜 History:")
        for b in ctx["history"]:
            print(f"  {b}")
        print("\n🌐 Geopolitical (2025-26):")
        for b in ctx["geopolitical"]:
            print(f"  {b}")
        print(f"\n📢 Impact: {ctx['impact_note']}")

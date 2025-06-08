import json

#starting stat depends on quality, i.e. "Name_of_Stat":[S_progression, A_progression, B_progression]
#maximum stat is starting stat*4 and with equidistant changes per level
#last index in array indicates "%"-Stat
MAIN_STATS = [
    ("ATK_flat", [79, 53, 26, False]),
    ("HP_flat", [550, 367, 183, False]),
    ("DEF_flat", [46, 31, 15, False]),
    ("ATK_percent", [7.5, 5, 2.5, True]),
    ("HP_percent", [7.5, 5, 2.5, True]),
    ("DEF_percent", [12, 8, 4, True]),
    ("CRIT Rate", [6, 4, 2, True]),
    ("CRIT DMG", [12, 8, 4, True]),
    ("Anomaly Proficiency", [23, 15, 8, False]),
    ("PEN Ratio", [6, 4, 2, True]),
    ("Physical DMG Bonus", [7.5, 5, 2.5, True]),
    ("Fire DMG Bonus", [7.5, 5, 2.5, True]),
    ("Ice DMG Bonus", [7.5, 5, 2.5, True]),
    ("Electric DMG Bonus", [7.5, 5, 2.5, True]),
    ("Ether DMG Bonus", [7.5, 5, 2.5, True]),
    ("Anomaly Mastery", [7.5, 5, 2.5, True]),
    ("Impact", [4.5, 3, 1.5, True]),
    ("Energy Regen", [15, 10, 5, True])
]
#corresponds to S, A, B tier
MAX_LEVELS = [("S", 15),("A", 12),("B", 9)]

#starting stat depends on quality, i.e. "Name_of_Stat":[S_progression, A_progression, B_progression]
#leveled stat is simply another stat added per roll, i.e. stat=roll*stat
#last index in array indicates "%"-Stat
#S has max 5 rolls in stat, A max 3 rolls, B max 1 roll
SUB_STATS = [
    ('ATK_flat', [19, 13, 6, False]),
    ('HP_flat', [112, 75, 37, False]),
    ('DEF_flat', [15, 10, 5, False]),
    ('ATK_percent', [3, 2, 1, True]),
    ('HP_percent', [3, 2, 1, True]),
    ('DEF_percent', [4.8, 3.2, 1.6, True]),
    ('CRIT Rate', [2.4, 1.6, 0.8, True]),
    ('CRIT DMG', [4.8, 3.2, 1.6, True]),
    ('Anomaly Proficiency', [9, 6, 3, False]),
    ('PEN', [9, 6, 3, False])
]
MAX_ROLL = [5, 3, 1]

# mapping
LEVEL_CLASS_MAPPING = {}
MAIN_CLASS_MAPPING = {}
SUB_CLASS_MAPPING = {}
LEVEL_REVERSE_LOOKUP = {} # optional
MAIN_REVERSE_LOOKUP = {}
SUB_REVERSE_LOOKUP = {}

#def serialize_key(stat_name, rolls, value):
#    return f"{stat_name}|{rolls}|{value}"
def add_to_level_mapping(tier:str, level: str, mapping: dict):
    key = f"{tier}|{level}"
    if key not in LEVEL_REVERSE_LOOKUP:
        class_id = len(mapping)
        mapping[str(class_id)] = {
            "tier": tier,
            "level": level
        }
        LEVEL_REVERSE_LOOKUP[key] = class_id

def add_to_main_mapping(stat_name: str, value: str, mapping: dict):
    #key = serialize_key(stat_name, rolls, value)
    key = f"{stat_name}|{value}"
    if key not in MAIN_REVERSE_LOOKUP:
        class_id = len(mapping)
        mapping[str(class_id)] = {
            "stat_name": stat_name,
            "value": value
        }
        MAIN_REVERSE_LOOKUP[key] = class_id 

def add_to_sub_mapping(stat_name: str, rolls: str, value: str, mapping: dict):
    #key = serialize_key(stat_name, rolls, value)
    key = f"{stat_name}|{rolls}|{value}"
    if key not in SUB_REVERSE_LOOKUP:
        class_id = len(mapping)
        mapping[str(class_id)] = {
            "stat_name": stat_name,
            "rolls": rolls,
            "value": value
        }
        SUB_REVERSE_LOOKUP[key] = class_id

def generate_level_classes():
    for i in MAX_LEVELS:
        for j in range(i[1]+1):
            tier = i[0]
            level = str(j)
            add_to_level_mapping(tier, level, LEVEL_CLASS_MAPPING)
    
    with open("mappings/level_class_mapping.json", "w", encoding="utf-8") as f:
        json.dump(LEVEL_CLASS_MAPPING, f, indent=2, ensure_ascii=False)

def generate_main_classes():
    for i in MAIN_STATS:
        for j in range(3):
            for x in range(MAX_LEVELS[j][1]+1):
                stat_name = i[0]
                #sadly, we cannot automatically infer the tier and level from main stat alone
                #ex: lvl2 A, lvl6 B, lvl0 S have the same PEN Ratio of 6%
                #tier = MAX_LEVELS[j][0]
                #level = str(x)
                stat_value_min = i[1][j]
                value = stat_value_min+((3*stat_value_min/MAX_LEVELS[j][1])*x)
                if i[1][3]:
                    value_text = str(int(value))+"%" if value == int(value) else f"{value:.1f}%"
                else:
                    value_text = str(int(value//1))
                #print(f"{stat_name} {value_text} {tier} {level}")
                add_to_main_mapping(stat_name, value_text, MAIN_CLASS_MAPPING)

    with open("mappings/mainstat_class_mapping.json", "w", encoding="utf-8") as f:
        json.dump(MAIN_CLASS_MAPPING, f, indent=2, ensure_ascii=False)       

def generate_sub_classes():
    for i in SUB_STATS:
        for j in range(3):
            for x in range(MAX_ROLL[j]+1):
                stat_name = i[0]
                roll_str = "+"+str(x) if x != 0 else ""
                stat_value = i[1][j]*(x+1)
                if i[1][3]:
                    stat_value = str(int(stat_value))+"%" if stat_value == int(stat_value) else f"{stat_value:.1f}%"
                else:
                    stat_value = str(int(stat_value//1))
                add_to_sub_mapping(stat_name, roll_str, stat_value, SUB_CLASS_MAPPING)
    add_to_sub_mapping("Set Effect", "", "", SUB_CLASS_MAPPING)

    with open("mappings/substat_class_mapping.json", "w", encoding="utf-8") as f:
        json.dump(SUB_CLASS_MAPPING, f, indent=2, ensure_ascii=False)


def main():
    generate_level_classes()
    generate_main_classes()
    generate_sub_classes()

if __name__ == "__main__":
    main()


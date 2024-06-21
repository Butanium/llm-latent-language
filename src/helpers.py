
lang_mapping = {
    'ja': 'japanese',
    'de': 'german',
    'en': 'english',
    'nl': 'dutch'
}

def get_colors(lang: str):
    colors = {
        'en': ["Red", "Green", "Blue", "Yellow", "Orange", "Purple", "Pink", "Brown", "Black", "White", "Gray", "Cyan", "Magenta", "Lime", "Olive", "Maroon", "Navy", "Teal", "Aqua", "Silver", "Gold"],
        'de': [
                "Rot", "Grün", "Blau", "Gelb", "Orange", "Lila", "Rosa", "Braun", "Schwarz", "Weiß",
                "Grau", "Cyan", "Magenta", "Limonengrün", "Oliv", "Weinrot", "Marineblau", "Türkis", "Aquamarin", "Silber",
                "Gold", "Beige", "Elfenbein", "Koralle", "Indigo", "Lavendel", "Mauve", "Ocker", "Pfirsich", "Perlmutt",
                "Rubinrot", "Smaragdgrün", "Saphirblau", "Schokolade", "Bronze", "Karminrot", "Schiefergrau", "Zinnoberrot", "Königsblau", "Himmelblau"
            ],
        'nl': [
                "Rood", "Groen", "Blauw", "Geel", "Oranje", "Paars", "Roze", "Bruin", "Zwart", "Wit",
                "Grijs", "Cyaan", "Magenta", "Limoengroen", "Olijf", "Bordeauxrood", "Marineblauw", "Turkoois", "Aqua", "Zilver",
                "Goud", "Beige", "Ivoor", "Koraal", "Indigo", "Lavendel", "Mauve", "Oker", "Perzik", "Parelmoer",
                "Robijnrood", "Smaragdgroen", "Saffierblauw", "Chocolade", "Brons", "Karmijnrood", "Leigrijs", "Vermiljoen", "Koningsblauw", "Hemelsblauw"
            ],
        'ja': []
    }
    colors = colors[lang]
    colors = [c.lower() for c in colors]
    return colors 

def get_tastes(lang: str):
    tastes = ["Sweet", "Sour", "Salty", "Bitter", "Umami", "Savory", "Spicy", "Astringent", "Metallic", "Minty", "Hot", "Bland", "Tastes good", "Tastes bad"]
    tastes = [t.lower() for t in tastes]
    return tastes 
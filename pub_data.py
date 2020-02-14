
class Pub:
    def __init__(self, name: str, distance, has_beer: bool=True, has_pub_quiz: bool=False,
is_spoons:bool = False, has_live_music=False, address:str= None, coordinates=None, has_funny_smell=False, **kwargs):
        """
        Plain Ol' Data class for a pub.
        :param name: The name of the pub (although in a dict we may end up storing this twice)
        :param distance: The walking distance in km as measured by Google Maps from the office.
        :param has_beer: Does this pub have real ale worth drinking?
        :param has_pub_quiz: Does this pub have a pub quiz on tonight?
        """
        self.name = name
        self.distance = distance
        self.has_beer = has_beer
        self.has_pub_quiz = has_pub_quiz
        self.is_spoons = is_spoons
        self.has_live_music = has_live_music
        self.address = address
        self.coordinates=coordinates
        self.has_funny_smell = has_funny_smell
        for key, val in kwargs.items():
            __setattr__(self, key, val)

    def __getattr__(self, attr):
        return None


PUBS = [Pub("The Bear Inn", 1.1, address="The Bear, 6, Alfred Street, Grandpont, Oxford, Oxfordshire, South East, England, OX1 4EH, United Kingdom", coordinates=(51.7515537, -1.2557349)),
        Pub("The Royal Blenheim", 1.3, address="Royal Blenheim, 13, St Ebbes Street, Grandpont, Oxford, Oxfordshire, South East, England, OX1 1PT, United Kingdom", coordinates= (51.7507949, -1.2593275)),
        Pub("The Chequers", 1.0, "The Chequers, 131, High Street, Grandpont, Oxford, Oxfordshire, South East, England, OX1 4DH, United Kingdom", coordinates=(51.759109, -1.1974538)),
	Pub("The Crown", 1.1, address="The Crown, 59a, Cornmarket Street, Grandpont, Oxford, Oxfordshire, South East, England, OX1 3HB, United Kingdom", coordinates= (51.7521883, -1.2580222)),
	Pub("The Eagle and Child", 0.65, "Eagle and Child, 49, St Giles', Norham Manor, Oxford, Oxfordshire, South East, England, OX1 3LU, United Kingdom",
coordinates= (51.757221099999995, -1.2603284108858848), has_funny_smell=True),
        Pub("The Four Candles", 1.1, is_spoons=True, address="The Four Candles, 51-53, George Street, Jericho, Oxford, Oxfordshire, South East, England, OX1 2BE, United Kingdom",
coordinates= (51.7533993, -1.2620909)),
        Pub("The Grapes", 1.0, address="The Grapes, 7, George Street, Grandpont, Oxford, Oxfordshire, South East, England, OX1 2AT, United Kingdom", coordinates= (51.7537486, -1.2595821)),
        Pub("The Head of the River", 1.7, address="The Head of the River, 1, Folly Bridge, Grandpont, Oxford, Oxfordshire, South East, England, OX1 4LB, United Kingdom",
coordinates= (51.746687800000004, -1.2560323412804122)),
        Pub("The Kings Arms", 0.6, address="King's Arms, 40, Holywell Street, Norham Manor, Oxford, Oxfordshire, South East, England, OX1 3SP, United Kingdom", coordinates=(51.7550797, -1.2543214)),
        Pub("The Lamb and Flag", 0.55, address="Lamb and Flag, St Giles', Norham Manor, Oxford, Oxfordshire, South East, England, OX1 3JS, United Kingdom", coordinates= (51.757404699999995, -1.2593104924183205), has_funny_smell=True),
        #Pub("The Mitre", 1.0, address="The Mitre, High Street, Grandpont, Oxford, Oxfordshire, South East, England, OX1 4AQ, United Kingdom", coordinates= (51.7523948, -1.2559797928120555)),
        Pub("The Old Tom", 1.2,address="Old Tom, 101, St Aldate\'s, Grandpont, Oxford, Oxfordshire, South East, England, OX1 1BT, United Kingdom", coordinates=(51.7508746, -1.2572208)),
        Pub("O'Neills", 1.0, has_beer=False, address="O'Neill's, George Street, Grandpont, Oxford, Oxfordshire, South East, England, OX1 2BJ, United Kingdom", coordinates=(51.7535341, -1.2607996)),
        Pub("The Oxford Retreat", 1.4,address="  The Oxford Retreat, 1-2, Hythe Bridge Street, Jericho, Oxford, Oxfordshire, South East, England, OX1 2TA, United Kingdom ",
coordinates= (51.7532923, -1.2656159)
),
        Pub("The Red Lion", 1.1,address="  The Red Lion, 40-42, Oxford Road, Old Marston, Oxford, Oxfordshire, South East, England, OX3 0PH, United Kingdom ",coordinates= (51.7723529, -1.2363012)

),
        Pub("The St Aldate's Tavern", 1.2,address="  St Aldate's Tavern, 108, St Aldate's, Grandpont, Oxford, Oxfordshire, South East, England, OX1 1BU, United Kingdom ",
coordinates= (51.75129325, -1.2574114966141754)
),
        Pub("The Three Goats Heads", 1.0,address="  Three Goats Heads, 3A, St Michael's Street, Grandpont, Oxford, Oxfordshire, South East, England, OX1 2DR, United Kingdom ",
coordinates= (51.7534232, -1.2590829)
),
        Pub("The Turf Tavern", 0.5,address="  Turf Tavern, 7, Bath Place, Grandpont, Oxford, Oxfordshire, South East, England, OX1 3SU, United Kingdom ",
coordinates= (51.7546944, -1.2528622320221636)
),
        Pub("The Wheatsheaf", 1.0,address="  The Wheatsheaf, 129, High Street, Grandpont, Oxford, Oxfordshire, South East, England, OX1 4DF, United Kingdom ",
coordinates= (51.75174, -1.256298)
),
        Pub("White Horse", 0.650, address="White Horse, 52, Broad Street, Grandpont, Oxford, Oxfordshire, South East, England, OX1 3BB, United Kingdom", coordinates=(51.7545992, -1.2556516)),
        Pub("The White Rabbit", 0.650, address="  The White Rabbit, 21, Friars Entry, Jericho, Oxford, Oxfordshire, South East, England, OX1 2BY, United Kingdom ",
coordinates= (51.7544715, -1.2607005706770802)
),
        Pub("The Castle", 1.6, address="  The Castle, 24, Paradise Street, Grandpont, Oxford, Oxfordshire, South East, England, OX1 1LD, United Kingdom ",
coordinates= (51.7507181, -1.2619326553382662)
),
        Pub("The Jam Factory", 1.6,address="  The Jam Factory, Hollybush Row, Jericho, Oxford, Oxfordshire, South East, England, OX1 1HU, United Kingdom ",
coordinates= (51.752345500000004, -1.2677711621899035)
),
        Pub("The Jolly Farmers", 1.5,address="  The Jolly Farmers, 20, Paradise Street, Grandpont, Oxford, Oxfordshire, South East, England, OX1 1LD, United Kingdom ",
coordinates= (51.750692349999994, -1.2626142092525714)
),
        Pub("The Swan and Castle", 1.5, is_spoons=True,address="  The Swan and Castle, 40, Castle Street Square, Oxford Castle Quarter, Grandpont, Oxford, Oxfordshire, South East, England, OX1 1AY, United Kingdom ",
coordinates= (51.7511807, -1.2618425)
),
        Pub("The Old Bookbinders", 1.5,address="  The Old Bookbinders, 17-18, Victor Street, Jericho, Oxford, Oxfordshire, South East, England, OX2 6BT, United Kingdom ",
coordinates= (51.7584939, -1.2697797)
),
        Pub("The Harcourt Arms", 1.4, has_pub_quiz=True,address="  The Harcourt Arms, 1-2, Cranham Terrace, Jericho, Oxford, Oxfordshire, South East, England, OX2 6DG, United Kingdom ",
coordinates= (51.75941005, -1.268024897384397)
),
        Pub("The Jericho Tavern", 1.3,address="  Jericho Tavern, 56, Walton Street, Jericho, Oxford, Oxfordshire, South East, England, OX2 6BU, United Kingdom ",
coordinates= (51.76017715, -1.2664924695983872)
),
        Pub("Jude the Obscure", 1.2,address="  Jude the Obscure, 51-54, Walton Street, Jericho, Oxford, Oxfordshire, South East, England, OX2 6AE, United Kingdom ",
coordinates= (51.7599843, -1.26628205555838)
),
        Pub("The Rickety Press", 1.4,address="  The Rickety Press, 67, Cranham Street, Jericho, Oxford, Oxfordshire, South East, England, OX2 6DS, United Kingdom ",
coordinates= (51.7598827, -1.2683244378576402)
),
        Pub("The Royal Oak", 0.9,address="  The Royal Oak, 42-44, Woodstock Road, Norham Manor, Oxford, Oxfordshire, South East, England, OX2 6HT, United Kingdom ",
coordinates= (51.7606495, -1.2616332397836085)
),
        Pub("The Victoria", 1.5,address="  The Victoria, 90, Walton Street, Jericho, Oxford, Oxfordshire, South East, England, OX2 6EB, United Kingdom ",
coordinates= (51.76173755, -1.2678698728609619)
),
        Pub("The Anchor", 2.0,address="  The Anchor, 2, Hayfield Road, Walton Manor, Oxford, Oxfordshire, South East, England, OX2 6TT, United Kingdom ",
coordinates= (51.7677013, -1.2690480938820095)
),
        Pub("The Gardener's Arms North Parade", 1.2,address="  Gardeners Arms, 8, North Parade Avenue, Norham Manor, Oxford, Oxfordshire, South East, England, OX2 6LX, United Kingdom ",
coordinates= (51.7644918, -1.2620828)
),
        Pub("The Gardener's Arms Plantation Road", 1.4,address="  The Gardeners Arms, 39, Plantation Road, Walton Manor, Oxford, Oxfordshire, South East, England, OX2 6JE, United Kingdom ",
coordinates= (51.76273625, -1.2665866949651243)
),
        Pub("The Rose and Crown", 1.2,address="  The Rose And Crown, 14, North Parade Avenue, Norham Manor, Oxford, Oxfordshire, South East, England, OX2 6LX, United Kingdom ",
coordinates= (51.7647059, -1.2616289)
),
        Pub("The Fishes", 3.4,address="  The Fishes, North Hinksey Village, North Hinksey, Vale of White Horse, Oxfordshire, South East, England, OX2 0NA, United Kingdom ",
coordinates= (51.7452124, -1.2825397)
),
        Pub("Tap Social Movement", 3.5, has_pub_quiz=True,address="  Tap Social Movement, 27, Southern By-pass Road, North Hinksey, Vale of White Horse, Oxfordshire, South East, England, OX2 0LX, United Kingdom ",
coordinates= (51.7512935, -1.295217)
),
        Pub("The Angel and Greyhound", 1.4,address="  Angel and Greyhound, 30, St Clements Street, New Marston, Oxford, Oxfordshire, South East, England, OX4 1AB, United Kingdom ",
coordinates= (51.75033915, -1.2426479795125314)
),
        Pub("The Half Moon", 1.3, address="  The Half Moon, 17-18, St Clements Street, New Marston, Oxford, Oxfordshire, South East, England, OX4 1AB, United Kingdom ",
coordinates= (51.7502867, -1.2434793)
),
        Pub("The Old Black Horse", 1.4, has_beer=False, address="Old Black Horse Inn, 102, St Clements Street, New Marston, Oxford, Oxfordshire, South East, England, OX4 1AR, United Kingdom", coordinates=(51.75001, -1.24279)),
        Pub("The Port Mahon", 1.6,address="Port Mahon, 82, St Clements Street, New Marston, Oxford, Oxfordshire, South East, England, OX4 1AW, United Kingdom", coordinates=(51.75042,-1.23997)
),
        Pub("The Star", 1.8, address="  The Star, 21, Rectory Road, New Marston, Oxford, Oxfordshire, South East, England, OX4 1BU, United Kingdom ",
coordinates= (51.7493995, -1.2382685933215969)
),
        Pub("The Black Swan", 2.0, address="  Black Swan, 11, Crown Street, Oxford, Oxfordshire, South East, England, OX4 1QG, United Kingdom ",
coordinates= (51.7466865, -1.2369835132352924)),
        Pub("The Cape of Good Hope", 1.3,address="  Cape of Good Hope, 1, Iffley Road, Grandpont, Oxford, Oxfordshire, South East, England, OX4 1EA, United Kingdom ",
coordinates= (51.74968525, -1.243796360612483)
),
        Pub("The Chester", 2.4, address="The Chester, 19, Chester Street, Cold Harbour, Oxford, Oxfordshire, South East, England, OX4 1SN, United Kingdom", coordinates=(51.7416405, -1.2406271)),
        Pub("The City Arms", 2.4, has_beer=False,address="  The City Arms, 288, Cowley Road, Oxford, Oxfordshire, South East, England, OX4 1UR, United Kingdom ",
coordinates= (51.744935299999995, -1.2301129178143755)
), 
        Pub("The Cowley Retreat", 2.0,address="  The Cowley Retreat, 172, Cowley Road, Oxford, Oxfordshire, South East, England, OX4 1UE, United Kingdom ",
coordinates= (51.747191799999996, -1.2355539858165292)
),
        Pub("BrewDog", 1.8,address="  Brewdog, 119, Cowley Road, Oxford, Oxfordshire, South East, England, OX4 1JH, United Kingdom ",
coordinates= (51.74807735, -1.2371606804715767)
),
        Pub("The Mad Hatter", 1.5, has_beer=False,address="  The Mad Hatter, 43, Iffley Road, Grandpont, Oxford, Oxfordshire, South East, England, OX4 1EA, United Kingdom ",
coordinates= (51.7484099, -1.2425581)
),
        Pub("The Fir Tree", 2.0, has_pub_quiz=True, address="The Fir Tree, 163, Iffley Road, Cold Harbour, Oxford, Oxfordshire, South East, England, OX4 1EJ, United Kingdom",
coordinates= (51.744434549999994, -1.2400197887848077)
),
        Pub("The James Street Tavern", 1.9, has_live_music=True,address="James Street Tavern, 47-48, James Street, Cold Harbour, Oxford, Oxfordshire, South East, England, OX4 1EU, United Kingdom",
coordinates= (51.7474273, -1.2373453741933642)
),
        Pub("The Library", 2.0, address="The Library, 182, Cowley Road, Oxford, Oxfordshire, South East, England, OX4 1UE, United Kingdom",
coordinates= (51.7470254, -1.2349056)),
        Pub("The Rusty Bicycle", 2.5, address="The Rusty Bicycle, 28, Magdalen Road, Oxford, Oxfordshire, South East, England, OX4 1RB, United Kingdom",
coordinates= (51.74293385, -1.234264793399705)
),
Pub("The University Club", 0.2, address="University Club, 11, Mansfield Road, Norham Manor, Oxford, Oxfordshire, South East, England, OX1 3SZ, United Kingdom",
coordinates= (51.757122800000005, -1.251335944893838)
),
Pub("The Up in Arms", 1.4, address="252-242 Marston Rd, Oxford", coordinates=(51.7594972,-1.2368538))]

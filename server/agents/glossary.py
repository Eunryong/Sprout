"""Plain-language glossary handed to the LLM agents so they can compose sensible scenes and prose.
It says WHAT things are (common sense), never which option is good: the rules table stays with the engine."""
from server.world.ontology import KINDS, PROBLEMS, VERBS

PROBLEM_GLOSS = {
    "THIRSTY": "Grandma is thirsty.",
    "FLOOR_WET": "Water has spilled on the floor.",
    "DARK": "The lights are out; it is pitch dark.",
    "COLD": "A cold draught; Grandma is shivering.",
    "VISITOR": "A little visitor has come to play.",
    "HUNGRY": "Grandma is hungry.", "UPSET": "A little one is upset and crying.", "MESSY": "Toys are scattered all over the floor.", "HOT": "The room is stuffy and hot.", "RAIN": "Rain is coming in through the window.",
    "BABY_HUNGRY": "The baby is hungry and fussing for milk.", "DIRTY_HANDS": "Hands are muddy after playing outside.",
    "HURT": "The child has scraped a knee and it stings.", "BORED": "The child is bored and has nothing to do.",
    "SCHOOL_MORNING": "It is school morning and time to get ready.", "FRIEND_SAD": "A friend at school is sad and sitting alone.",
}
VERB_GLOSS = {
    "DRINK": "drink from the target", "WIPE": "wipe the floor with the target", "LIGHT": "switch the target on",
    "COVER": "cover Grandma (or a scraped knee) with the target", "GIVE": "give the target to someone (the visitor, a sad friend)",
    "FETCH": "bring the target to Grandma", "WAIT": "wait and do nothing (no target)",
    "EAT": "eat the target", "HUG": "hold the target close to comfort the little one", "TIDY": "put the toys away in the target", "FAN": "fan Grandma with the target", "SHELTER": "hold the target over everyone against the rain",
    "FEED": "feed the baby with the target", "WASH": "wash dirty hands with the target", "DRAW": "draw with the target",
    "READ": "read the target together", "WEAR": "put the target on",
}
KIND_GLOSS = {
    "cup": "a drinking cup", "cloth": "a piece of cloth", "paper": "a sheet of paper", "blanket": "a wool blanket",
    "flashlight": "an electric torch", "spoon": "a metal spoon", "ball": "a rubber ball",
    "bread": "a round loaf of bread", "teddy": "a soft plush teddy", "box": "a wooden toy box", "fan": "a plastic hand fan", "umbrella": "a big cloth umbrella",
    "bottle": "a baby's milk bottle", "soap": "a bar of soap", "bandage": "a small sticking bandage", "crayon": "a wax crayon",
    "book": "a picture book", "backpack": "a canvas school backpack",
}


# The first stage (the baby cannot choose yet): the baby is the one cared for, so nothing is done for Grandma.
BABY_GLOSS = {"BABY_HUNGRY": PROBLEM_GLOSS["BABY_HUNGRY"], "UPSET": "The baby is crying and cannot settle.", "COLD": "The baby is cold and whimpering."}
BABY_VERB_GLOSS = {
    "FEED": "feed the baby with the target", "COVER": "wrap the baby in the target", "GIVE": "give the baby the target to hold",
    "HUG": "cuddle the baby with the target", "FETCH": "bring the baby the target", "WIPE": "wipe the baby's face with the target",
    "DRINK": "let the baby drink from the target", "EAT": "give the baby the target to nibble", "LIGHT": "switch the target on near the baby",
    "READ": "show the baby the target", "DRAW": "wave the target in front of the baby", "WASH": "wash the baby's hands with the target",
    "WEAR": "dress the baby in the target", "SHELTER": "hold the target over the baby", "TIDY": "put the target away",
}


# From 12 months to 5 years the child's troubles are its own (voice "self"): thirsty, cold and hungry are the child's, not Grandma's.
SELF_GLOSS = {"THIRSTY": "The child is thirsty.", "COLD": "A cold draught; the child is shivering.", "HUNGRY": "The child is hungry."}
SELF_VERB_GLOSS = {"COVER": "wrap up in the target", "HUG": "hold the target close", "GIVE": "hand over the target", "FETCH": "go and get the target",
                   "FEED": "eat with the target"}


def problem_gloss(tag: str, voice: str = "grandma") -> str:
    return {"baby": BABY_GLOSS, "self": SELF_GLOSS}.get(voice, {}).get(tag) or PROBLEM_GLOSS[tag]


def verb_gloss(verb: str, voice: str = "grandma") -> str:
    return {"baby": BABY_VERB_GLOSS, "self": SELF_VERB_GLOSS}.get(voice, {}).get(verb) or VERB_GLOSS[verb]


def check_complete() -> None:
    assert set(PROBLEMS) <= set(PROBLEM_GLOSS) and set(VERBS) <= set(VERB_GLOSS) and set(KINDS) <= set(KIND_GLOSS)

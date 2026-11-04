"""Hand-built situations for engine tests (the generator is tested separately)."""
from server.world.ontology import KINDS
from server.world.situation import Obj, OptionSpec, Situation


def obj(kind, color="red", size="big") -> Obj:
    k = KINDS[kind]
    return Obj(kind=kind, color=color, shape=k.shape, size=size if size in k.sizes else k.sizes[0], material=k.material)


def sit(sid, tags, objs, opts, **kw) -> Situation:
    """objs: [(kind, color, size)]; opts: [(verb, object index | None)]"""
    return Situation(id=sid, tags=list(tags), objects=[obj(*o) for o in objs],
                     options=[OptionSpec(verb=v, target=t) for v, t in opts], **kw)


def wet(sid="w", **kw) -> Situation:
    # 0 WIPE cloth = solved(2) | 1 WIPE paper = partial(1) | 2 WAIT = worse(-1) | 3 DRINK cup = nothing(0)
    return sit(sid, ["FLOOR_WET"], [("cloth", "red", "big"), ("paper", "blue", "small"), ("cup", "green", "small")],
               [("WIPE", 0), ("WIPE", 1), ("WAIT", None), ("DRINK", 2)], **kw)


def thirsty(sid="t", color="blue", **kw) -> Situation:
    # 0 DRINK cup = solved | 1 DRINK spoon = nothing | 2 WIPE cup = nothing
    return sit(sid, ["THIRSTY"], [("cup", color, "small"), ("spoon", "red", "big"), ("ball", "green", "small")],
               [("DRINK", 0), ("DRINK", 1), ("WIPE", 0), ("WAIT", None)], **kw)


def dark(sid="d", **kw) -> Situation:
    return sit(sid, ["DARK"], [("flashlight", "red", "big"), ("spoon", "blue", "small"), ("cup", "green", "big")],
               [("LIGHT", 0), ("LIGHT", 1), ("DRINK", 2)], **kw)


def visitor(sid="v", **kw) -> Situation:
    return sit(sid, ["VISITOR"], [("ball", "red", "big"), ("cup", "blue", "small"), ("cloth", "green", "small")],
               [("GIVE", 0), ("GIVE", 1), ("WIPE", 2), ("WAIT", None)], **kw)


def cold(sid="c", **kw) -> Situation:
    return sit(sid, ["COLD"], [("blanket", "red", "big"), ("cloth", "blue", "small"), ("ball", "green", "small")],
               [("COVER", 0), ("COVER", 1), ("GIVE", 2), ("WAIT", None)], **kw)


def request(sid, kind, word, token, **kw) -> Situation:
    others = [k for k in ("cup", "cloth", "paper", "ball") if k != kind][:2]
    return sit(sid, ["REQUEST"], [(kind, "red", "big"), (others[0], "blue", "small"), (others[1], "green", "small")],
               [("FETCH", 0), ("FETCH", 1), ("FETCH", 2)], word=word, word_token=token, word_kind=kind, **kw)


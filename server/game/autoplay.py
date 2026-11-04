"""Synthetic player (a script that knows the engine truth). For smoke tests and demos, never for the child."""
import random

from server.world.state import Phase

from .session import Session


def play_day(s: Session, rng: random.Random, p_show: float = 0.5) -> dict:
    first = True
    while s.state.phase != Phase.EVENING:
        if s.state.phase == Phase.CARDS:
            s.draw_cards()
            s.view()  # the player looks at the cards (and so the narrator is asked, as in the real game)
            s.pick_card(rng.randrange(2))
            s.view()
        best = s.debug_best()
        if (first and s.state.day == 1) or rng.random() < p_show:
            s.show(rng.choice(best))
        else:
            s.try_child()
        first = False
    return s.end_day()

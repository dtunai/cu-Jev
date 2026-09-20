# Examples

## Starfighter — a browser game flown by typed decisions

```bash
uv run cujev serve --model models/Qwen3.5-4B --port 8080
# open http://127.0.0.1:8080/play   (add ?autostart=1 to skip the Start button)
```

`examples/starfighter/` is a plain HTML/JS canvas game served by the cu-Jev
server itself (no build step). Every tick the game writes what it sees as
text and asks four typed questions in one `/v1/systemone` request:

| question | type | criteria (built from the live game state) |
|---|---|---|
| `move` | choice | the current lane and its neighbours (`lane 2` / `lane 3` / `lane 4`), each described with what is in it — *"asteroid wall 34u, 5 credits 52u (bonus) (risky)"* — and a literal tag: *safe*, *risky*, *DANGER: hit within a second*, *far, shootable* |
| `fire` | noul | is a fighter, asteroid or wall ahead in the ship's own lane? |
| `shield` | noul | does the state tag the ship's own lane with the word DANGER? |
| `threat` | score | calm / caution / danger / critical → the HUD meter |

What makes it a game of decisions rather than reflexes: every ~8 s an
**asteroid wall** closes four of the five lanes and the pilot has to read
which lane is the gap and get there; **credit** clusters appear in random
lanes; enemy fighters shoot and **drift between lanes** while still far away,
so the state always shows them coming. The ship moves one lane per tick, and
the pilot only ever chooses between staying and stepping one lane, so the path
is never hidden from it.

The answers go through visible game rules (lane edges, gun cooldown, shield
charge) and the probabilities are drawn live in the panel. "Pilot orders" is
a free-text field appended to every state — edit it while playing and watch
the policy change. Tick "I fly" to take the controls yourself (← → space,
shift = shield).

Numbers on an RTX 3090 with Qwen3.5-4B: ~95 ms per decision (≈ 500 tokens of
state + four questions, prefilled fresh every tick because the state changes),
≈ 11 decisions/s. With Qwen3.5-0.8B it is ≈ 30 ms and 30 decisions/s.

`?api=http://host:port` points the page at a server on another origin (CORS
is open); `?model=` picks the served model name; `?policy=` presets the orders.

`assets/starfighter.gif` / `.mp4` were recorded from this page with
`record.py` (Playwright, headless Chromium, `--use-gl=swiftshader`, 1180×720)
on the 4B: three consecutive autoplayed 40 s takes scored 1724 / 1051 / 335
with 0 / 1 / 1 hull losses; the GIF is the first one. The pilot is not
perfect — it dies about once a minute, mostly to bullets fired from a lane it
is stepping through — and that is the honest picture of a 4B reading a text
state at ~11 decisions per second.

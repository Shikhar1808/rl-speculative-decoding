# The race demo

```bash
uv run python -m specrl.app                    # SmolLM2 pair -> http://localhost:7860
uv run python -m specrl.app --share-lan        # also reachable from the other laptop: http://<this-PC-IP>:7860
uv run python -m specrl.app --pair qwen
```

Needs the trained policies for that pair (`policies/<pair>_*.pt`).

## What the audience sees

| Element | Meaning |
|---|---|
| Left pane | target model alone, one token per big-model call |
| Right pane | target + chosen policy. Green = accepted draft, red struck-through = rejected draft, **bold** = target's own token, blue bar = where the agent pressed STOP. Hover a token for its draft confidence |
| Confidence meter | the latest round's draft confidences; green bars accepted, red rejected |
| Summary | tokens/s of each side, speedup, target calls saved, whether outputs are identical |
| Policy dropdown | the four RL agents and two baselines, to compare live |

## How the race is timed (be ready to explain this)

The two generations run **one after the other** on the GPU, and every token gets a real timestamp. The
panes then replay both at the recorded speed, side by side. Running both at the same time would make them
share one GPU and slow each other down, which would not be a fair race. The numbers shown are the real
measured ones.

## Demo-day checklist

1. Plug in the laptop, close other GPU apps.
2. Start the app 5 minutes early (model loading + warm-up).
3. Run each example once (first run is slower).
4. Backup: screen-record a run (Win + Alt + R with Xbox Game Bar), and have the app ready on the second laptop.

Talk order (from the guide): race demo first → analogy → speculative decoding picture → MDP → four
algorithms → results → what the agent learned → audience prompt.

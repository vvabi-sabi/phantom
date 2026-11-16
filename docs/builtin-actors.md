# Bundled builtin actors

Phantom-animate ships four bundled procedural actors, generated locally from
original primitives (license: `BUILTIN-ACTORS-LICENSE.txt`, MIT). Nothing is
downloaded. Profile actions and defaults are shared by GUI, CLI and MCP through
one catalog (`crates/app/src/builtin.rs`).

| Actor     | Actions                          | Default |
|-----------|----------------------------------|---------|
| humanoid  | idle, walk, jump                 | idle    |
| wolf      | idle, walk, sit, lie, sneak      | idle    |
| horse     | idle, trot, gallop               | idle    |
| bird      | fly, perch                       | perch   |

Keys with an action unsupported by the actor are rejected atomically
(GUI/CLI/MCP all go through the same command dispatch).

## GUI (`phantom-animate edit <project>`)

- Right panel: pick an actor in the "Actor" combo, then "Add Phantom".
- Timeline key row (with a phantom selected): XYZ/yaw fields plus an action
  combo listing only the selected actor's actions. Imported rigs show "(clip)"
  and stay clip-driven.

## CLI

```sh
phantom-animate add-phantom <project> --scene 1 <name> --rig wolf
# --rig: builtin (legacy humanoid) | humanoid | wolf | horse | bird | <asset-id>
phantom-animate set-key <project> --scene 1 --phantom <id> --frame 0 \
    --pos 0,0,0 --yaw 0 --action sneak
```

`set-key` accepts all catalog actions; the command fails if the actor does not
support the action.

## MCP

- `phantom_add { scene, name, rig_actor? , rig_asset? }` — `rig_actor` is one of
  `humanoid|wolf|horse|bird`; omitting both keeps the legacy builtin humanoid;
  `rig_asset` (imported FBX) is unchanged. Passing both is an error.
- `builtin_catalog {}` — returns profiles with `actions` and `default_action`.
- `key_set` and `take_add` accept any catalog action name.

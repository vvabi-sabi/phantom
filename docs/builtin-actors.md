# Bundled builtin actors

Phantom-animate ships seven bundled procedural actors, generated locally from
original primitives (license: `BUILTIN-ACTORS-LICENSE.txt`, MIT). Nothing is
downloaded. Profile poses and defaults are shared by GUI, CLI and MCP through
one catalog (`crates/app/src/builtin.rs`).

Actors are not animated: a key holds one pose, and the pose switches at its own
key. Each species declares the poses it can hold.

| Actor     | Poses            | Default |
|-----------|------------------|---------|
| humanoid  | stand, sit, lie  | stand   |
| child     | stand, sit, lie  | stand   |
| teen      | stand, sit, lie  | stand   |
| elder     | stand, sit, lie  | stand   |
| wolf      | stand, sit, lie  | stand   |
| horse     | stand, lie       | stand   |
| bird      | stand, sit       | stand   |

`child`, `teen` and `elder` are not separate rigs: they share the humanoid
skeleton and its poses, and only the proportion profile differs — taken from
`HUMAN_ROLES` (`adult|child|teen|elder`), the same table that gives the
reference eye/shoulder/hip/knee lines. A child is not a shrunk adult: legs are
short relative to the torso, so regions scale separately, and the adult stays
the identity scale (its geometry did not change when the roles were added).

Animals in the prop catalog (`cat`, `dog_big`, `cow`, `deer`, …) are props, not
actors: no skeleton, no poses of their own. They move and turn through the
object's transform keys, and a pose variant is a catalog entry
(`cat_sitting`, `dog_big_lying`).

A pose the actor does not have is rejected atomically with `unsupported_pose`
(GUI/CLI/MCP all go through the same command dispatch) — a horse is never
silently seated. An imported rig has no poses: it renders in bind pose and any
pose on its key is ignored.

## GUI (`phantom-animate edit <project>`)

- Right panel: pick an actor in the "Actor" combo, then "Add Phantom".
- Timeline key row (with a phantom selected): XYZ/yaw fields plus a pose combo
  listing only the selected actor's poses. Imported rigs show "(bind pose)" and
  stay where the file put them.

## CLI

```sh
phantom-animate add-phantom <project> --scene 1 <name> --rig wolf
# --rig: builtin (legacy humanoid) | humanoid | child | teen | elder | wolf | horse | bird | <asset-id>
phantom-animate set-key <project> --scene 1 --phantom <id> --frame 0 \
    --pos 0,0,0 --yaw 0 --pose sit
```

`set-key` accepts all catalog poses; the command fails if the actor does not
have the pose. `--yaw` is in degrees.

## MCP

- `phantom_add { scene, name, rig_actor? , rig_asset? }` — `rig_actor` is one of
  `humanoid|child|teen|elder|wolf|horse|bird`; omitting both keeps the legacy builtin humanoid;
  `rig_asset` (imported FBX) is unchanged. Passing both is an error.
- `builtin_catalog {}` — returns profiles with `poses`, `default_pose`, the
  metric prop table, human reference lines and the stage presets.
- `key_set` and `motion_replace` accept any catalog pose name (`stand|sit|lie`);
  `key_set` preserves an omitted pose from the exact key or the evaluated state.

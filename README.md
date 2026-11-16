# Phantom-animate

Stage short stylized animation by directing 3D characters like actors — without the complexity of a full Blender setup. The director stages a scene, puts the background on a curved arc screen, imports characters and motion (OBJ/FBX, Mixamo-compatible), drives Phantom from the keyboard or from keys on the timeline, records takes, sets up the camera, and gets frames with 2D styling (inverted-hull outline).

The full design and status document is [docs/architecture.md](docs/architecture.md); the bundled actors and their actions are listed in [docs/builtin-actors.md](docs/builtin-actors.md).

## How the program is put together

A workspace of three crates, with dependencies pointing `app → runtime → core`:

| Crate | Role |
|---|---|
| `crates/core` (`phantom-core`) | Project document (serde/JSON), commands with undo, 24 fps timeline, Phantom, takes, camera, scene state evaluation, audio mixing. No GPU/Bevy/window dependencies. |
| `crates/runtime` (`phantom-runtime`) | Project file storage, OBJ/FBX import via ufbx, FFmpeg (video/audio decode, MP4 export), cpal audio, Bevy GPU rendering (headless and viewport), export jobs. |
| `crates/app` (`phantom-animate`) | The binary: CLI, GUI editor (bevy + bevy_egui), MCP server (rmcp, stdio), and the bridge to an open window (Unix socket). |

Both surfaces (GUI and MCP) call one typed-command dispatcher; the UI never mutates the document directly. A project on disk is a directory containing `manifest.json` and an `assets/` folder.

Key concepts: **scene** (a frame range at 24 fps), **Phantom** (a controllable skeleton with an attached model), **background** (a curved elliptical arc screen: image or video; screen zones light the scene), **take** (a recorded performance — the active take overrides the manual track within its range), **film** (a sequence of clip links pointing at scene ranges).

Coordinates: right-handed system, Y up, meters. Phantom actions: `idle`, `walk`, `jump`.

## Requirements

- **Linux** (the only target platform), **a Vulkan-capable GPU** for preview and rendering (headless render through Bevy/WGPU). Editing a project without a GPU is possible, but not guaranteed for every command.
- **Rust** (stable, edition 2021).
- **FFmpeg/ffprobe** on `PATH` — only for video backgrounds, audio, and MP4 export (H.264 needs a build with `libx264`). PNG export and everything else work without it.

## Build

```sh
cargo build --release
# binary: target/release/phantom-animate
```

Tests (one GPU test is marked `ignored`):

```sh
cargo test --release
cargo test --release -- --ignored   # tests that require a GPU
```

Some `runtime` and `app` tests read test assets from the workspace root
(`../../Hip Hop Dancing.fbx`, `../../43-obj/obj/Wolf_obj.obj`). These are
third-party models and are not checked into the repository: on a clean clone
those tests fail with `NotFound`, and the rest pass. To run them too, put the
FBX clip in the workspace root and the OBJ model in `43-obj/obj/`.

## Running: CLI

Every editing command works on a project directory and saves it atomically. Rendering requires a GPU.

```sh
PA=target/release/phantom-animate

# 1. Create a project (a directory with manifest.json + assets/)
$PA new ./my-project "Toon"

# 2. Add a scene (frame range is half-open [start, end), 0-96 by default)
$PA add-scene ./my-project "Scene 1" --frames 0-96

# 3. Import assets: obj/fbx — models and clips, png/jpg — images,
#    mp3/wav/ogg/flac/m4a — audio, mp4/gif/webm/mov — video backgrounds
$PA import ./my-project ./Wolf_obj.obj
$PA import ./my-project ./character.fbx
$PA import ./my-project ./background.png

# add-scene/import print the scene and asset IDs (one counter for the whole project, starting at 1)
# 4. Background on the arc screen
$PA set-background ./my-project --scene <scene-id> --asset <background-image-id> --center 0,2,-5 --arc 2.0 --radius 6.0 --height 4.0
#    video background: --video --asset <video-id>

# 5. Phantom: the builtin humanoid rig or an imported FBX character
$PA add-phantom ./my-project "Hero" --scene <scene-id> --rig builtin
$PA add-phantom ./my-project "Wolf" --scene <scene-id> --rig <fbx-asset-id>

# 6. Phantom keys (position x,y,z, action) and camera
#    pass values that start with "-" using "=": --pos=-2,0,0
$PA set-key ./my-project --scene <scene-id> --phantom <phantom-id> --frame 0 --pos=-2,0,0 --action walk
$PA set-key ./my-project --scene <scene-id> --phantom <phantom-id> --frame 48 --pos 2,0,0 --action idle
$PA set-camera ./my-project --scene <scene-id> --pos 0,1.4,4 --focal 35

# 7. FBX motion clip on a rig, audio
$PA set-clip ./my-project --scene <scene-id> --phantom <phantom-id> --asset <clip-id>
$PA track-add ./my-project "music" --scene <scene-id>
$PA clip-add ./my-project --track <track-id> --asset <audio-id> --start 0 --offset 0 --len 230400

# 8. Cut the film and render it
$PA film-add ./my-project --scene <scene-id> --frames 0-96
$PA render ./my-project --scene <scene-id> --out ./frames --width 1280 --height 720   # PNG sequence
$PA render-film ./my-project --out ./film.mp4                                 # MP4 H.264 (+AAC when there is audio)
```

Full list: `phantom-animate --help` and `--help` on each subcommand. Rendering on twos is the `--twos` flag on `render`.

## Running: GUI editor

```sh
target/release/phantom-animate edit ./my-project
```

A storyboarding window (bevy + bevy_egui): a viewport rendering the scene, a timeline with keys/scrub/play, an outliner, 2D drawing on top of the frame (brush/eraser/color/size, onion skin), and performance recording. Phantom control is active while the cursor is over the viewport: **WASD** — move, **Space** — jump; recording runs from the current frame as one atomic command. PNG export from the window is launched as a separate CLI process (GPU isolation).

## MCP server

The MCP server (rmcp, stdio protocol) is meant for AI agents. `stdout` carries the protocol only, logs go to `stderr`. Two modes:

### 1. Standalone session (no window)

Creates its own session without a UI — for agent-driven project creation and rendering:

```sh
target/release/phantom-animate mcp
```

### 2. Bridge to an open editor

While `edit` is running it writes a lock file `.phantom.lock` into the project directory and keeps a Unix socket in the user's private directory (0700). The bridge MCP server connects to that socket, so the agent works on the very project the user sees open (while a scene recording is live, mutation commands are rejected as busy):

```sh
# terminal 1: open the project in a window
target/release/phantom-animate edit ./my-project
# terminal 2: bridge
target/release/phantom-animate mcp --bridge ./my-project
```

### Connecting an MCP client

Example configuration (Claude Desktop / Zed and other clients with a stdio transport):

```json
{
  "mcpServers": {
    "phantom-animate": {
      "command": "/absolute/path/to/phantom-animate",
      "args": ["mcp"]
    }
  }
}
```

### Available MCP tools

- **Project**: `project_new`, `project_open`, `project_save`, `state_get` (revision, scenes, assets, film length)
- **Scene and assets**: `scene_add`, `asset_import`, `background_set`
- **Phantom**: `phantom_add` (rig — builtin or FBX asset), `key_set`, `phantom_clip_set`, `take_add`, `take_active`, `take_remove`
- **Camera**: `camera_set` (frame, position, yaw/pitch in radians, focal length in mm)
- **Drawing**: `brush_stroke` (2D layer, points in canvas coordinates 1920×1080)
- **Film**: `film_add`, `film_move`, `film_trim`, `film_remove`
- **Audio**: `audio_track_add`, `audio_clip_add`, `track_volume_set`, `track_mute_set`, `audio_clip_remove`
- **Export (jobs)**: `render_scene` (PNG), `render_film` (MP4), `job_status`, `job_cancel`

Render jobs run as child CLI processes: `render_*` returns a job ID, progress and completion are read through `job_status`, cancellation goes through `job_cancel`. Frames are integers at 24 fps, `[start, end)` is half-open, actions are `idle|walk|jump`, coordinates are in meters with Y up. Long-running import/bake/export calls return a job ID and support cancellation.

## Repository

- `crates/core` — project document, commands, timeline, audio mix (no GPU, no window)
- `crates/runtime` — storage, OBJ/FBX import, FFmpeg, audio, GPU rendering
- `crates/app` — the binary: CLI, GUI, MCP server, and the bridge to an open window
- `docs/architecture.md` — architecture, decisions and implementation status
- `docs/builtin-actors.md` — the bundled procedural actors and their actions
- `BUILTIN-ACTORS-LICENSE.txt` — license for the bundled actors (MIT)

The Phantom Storyboard Blender add-on is a separate repository:
<https://github.com/vvabi-sabi/add_ons_Phantom> (a staging reference, not part of this build).

## License

MIT OR Apache-2.0 (see `Cargo.toml`). Third-party asset and codec licenses (FFmpeg/libx264) are reviewed separately before delivery.

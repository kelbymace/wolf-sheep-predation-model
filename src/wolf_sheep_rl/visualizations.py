import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

def animate_model(model, steps=200, interval=150):
    fig, ax = plt.subplots(figsize=(6, 6))

    grass_img = ax.imshow(
        model.patch_array(),
        origin="lower",
        interpolation="nearest",
        cmap="YlGn",
        vmin=0,
        vmax=1
    )

    sheep_scatter = ax.scatter([], [], s=25, marker="o", label="Sheep")
    wolf_scatter = ax.scatter([], [], s=40, marker="s", label="Wolves")

    title = ax.set_title("")
    ax.set_xlim(-0.5, model.width - 0.5)
    ax.set_ylim(-0.5, model.height - 0.5)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.legend(loc="upper right")

    anim = None

    def draw_current_state():
        grass_img.set_data(model.patch_array())

        sheep_xy = np.array([(s.x, s.y) for s in model.sheep]) if model.sheep else np.empty((0, 2))
        wolf_xy = np.array([(w.x, w.y) for w in model.wolves]) if model.wolves else np.empty((0, 2))

        sheep_scatter.set_offsets(sheep_xy)
        wolf_scatter.set_offsets(wolf_xy)

        title.set_text(
            f"Tick {model.ticks} | Sheep: {model.count_sheep()} | Wolves: {model.count_wolves()}"
        )

        return grass_img, sheep_scatter, wolf_scatter, title


    def update(frame):
        artists = draw_current_state()

        if frame < steps - 1:
            running = model.go()
            if not running and anim is not None:
                anim.event_source.stop()

        return artists

    anim = FuncAnimation(
        fig,
        update,
        frames=steps,
        interval=interval,
        blit=False,
        repeat=False
    )

    return anim
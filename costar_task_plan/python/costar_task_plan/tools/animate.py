import pygame as pg

'''
Animate planning results
'''


def animate(env, path, hz=10, loop=True, *args, **kwargs):
    try:
        clock = env.clock
        screen = env._world.getScreen()
        idx = 0
        while True:
            if path[idx].world is not None:
                path[idx].world.show(screen)
                if hz > 0:
                    clock.tick(hz)
                if not loop:
                    break
            idx = (idx + 1) % len(path)
    except KeyboardInterrupt, e:
        pass
    finally:
        pg.quit()

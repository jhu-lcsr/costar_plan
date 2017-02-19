
"""
(c) 2016 Chris Paxton
"""

import world as W
from world import *
from actor import *
import curses


class Graphics:

    def __init__(self):
        pass

    def drawWorld(self, world):
        pass

    def wait(self):
        pass

    def close(self):
        pass


class TerminalGraphics(Graphics):

    def __init__(self):
        self.stdscr = curses.initscr()
        curses.start_color()
        curses.init_pair(1, curses.COLOR_RED, curses.COLOR_BLACK)
        self.bottom_row = 0

    def wait(self):
        self.stdscr.addstr(self.bottom_row + 2, 0, "Press any key...")
        self.stdscr.getch()

    def getChar(self):
        self.stdscr.addstr(self.bottom_row + 2, 0, "Input command: ")
        return self.stdscr.getch()

    def write(self, x, y, string, settings=None):
        if not settings is None:
            self.stdscr.addstr(x, y, string, settings)
        else:
            self.stdscr.addstr(x, y, string)

    def writeLine(self, y, string, settings=None):
        if not settings is None:
            self.stdscr.addstr(self.bottom_row + y, 0, string, settings)
        else:
            self.stdscr.addstr(self.bottom_row + y, 0, string)

    def drawWorld(self, world, draw_actors=True):
        for i in range(world.worldmap.shape[0]):
            for j in range(world.worldmap.shape[1]):
                if world.worldmap[i, j] == W.DirectionEast:
                    self.stdscr.addstr(i, j, '>', curses.color_pair(0))
                elif world.worldmap[i, j] == W.DirectionWest:
                    self.stdscr.addstr(i, j, '<', curses.color_pair(0))
                elif world.worldmap[i, j] == W.DirectionNorth:
                    self.stdscr.addstr(i, j, '^', curses.color_pair(0))
                elif world.worldmap[i, j] == W.DirectionSouth:
                    self.stdscr.addstr(i, j, 'v', curses.color_pair(0))
                elif world.worldmap[i, j] == W.Sidewalk:
                    self.stdscr.addstr(i, j, '#', curses.color_pair(0))
                elif world.worldmap[i, j] == W.Intersection:
                    self.stdscr.addstr(i, j, 'X', curses.color_pair(0))
                else:
                    self.stdscr.addstr(i, j, ' ')

        if draw_actors:
            for actor in world.actors:
                if actor.state.x >= 0 and actor.state.y >= 0:
                    self.stdscr.addstr(
                        actor.state.y, actor.state.x, actor.name, curses.color_pair(1) + curses.A_BOLD + curses.A_UNDERLINE)

        self.bottom_row = i

    def close(self):
        curses.endwin()


class PlayerActor(Actor):

    def __init__(self, terminalGraphics, state,
                 name="P"):
        super(PlayerActor, self).__init__(state, name)
        self.impatience = 0
        self._tg = terminalGraphics

    def chooseAction(self, world):

        idx = (self._tg.getChar()) - 49
        self._tg.stdscr.addstr(
            self._tg.bottom_row + 2, 0, "Input command: %d" % (idx))
        if idx >= 0 and idx <= len(self.actions):
            return self.actions[idx]
        else:
            return self.actions[0]

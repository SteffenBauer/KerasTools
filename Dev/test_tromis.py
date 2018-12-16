#!/usr/bin/env python
# -*- coding: utf-8 -*-

import tromis

game = tromis.Tromis(width=3, height=5)
print game.get_state()
game.reset()
print game.piece
print game.get_state()


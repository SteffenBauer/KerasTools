#!/usr/bin/env python
# -*- coding: utf-8 -*-

import tromis

game = tromis.Tromis(width=3, height=5)
for i in range(10):
    game.reset()
    print game.get_state()


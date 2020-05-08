import io

import ipywidgets
import PIL, PIL.Image
import time

class GameGUI():
    actionMap = {'left': 'arrow-left',
                 'right': 'arrow-right', 
                 'up': 'arrow-up', 
                 'down': 'arrow-down', 
                 'forward': 'arrow-up', 
                 'rotateleft': 'rotate-left', 
                 'rotateright': 'rotate-right', 
                 'skip': 'dot-circle-o'}
    blay = ipywidgets.Layout(width='34px', height='30px', border='1px solid black')
    ilay = ipywidgets.Layout(width='80px', height='30px', border='1px solid black')
    scoreFmt = "<center><b>{:2.2f} / {:2.2f}</b></center>"
    
    def __init__(self, game):
        self.game = game
        self.actions = {self.actionMap[v]: k for k,v in self.game.actions.items()}
        initScore = self.game.get_score()
        self.aggScore = initScore
        
        bnew = ipywidgets.Button(layout=self.ilay, style=ipywidgets.ButtonStyle(font_weight='bold', button_color='green'), description = 'New')
        self.stat = ipywidgets.HTML(layout=self.ilay, value=self.scoreFmt.format(initScore, self.aggScore))
        
        controls = [bnew]
        for _, i in sorted(tuple(game.actions.items())):
            button = ipywidgets.Button(layout=self.blay, style=ipywidgets.ButtonStyle(font_weight='bold', button_color='yellow'), icon=self.actionMap[i])
            controls.append(button)
        for c in controls:
            c.on_click(self._onButtonClicked)
        controls.append(self.stat)
        self.ctrlbox = ipywidgets.HBox(controls)
                
        self.canvas = ipywidgets.Image()
        self.imbuf = io.BytesIO()
        self._plotGame(self.game.get_frame())
                
        ibox = ipywidgets.VBox([self.canvas, self.ctrlbox])
        ibox.layout.align_items = 'center'
        self.gamebox = ipywidgets.HBox([ibox])

    def _onButtonClicked(self, args):
        if args.description == 'New':
            self.game.reset()
            self.stat.value = ""
            self.aggScore = 0.0
        elif not self.game.is_over():
            args.style.button_color = 'red'
            time.sleep(0.1)
            args.style.button_color = 'yellow'
            self.game.play(self.actions[args.icon])
        else:
            return
        currentScore = self.game.get_score()
        self.aggScore += currentScore
        self.stat.value = self.scoreFmt.format(currentScore, self.aggScore)
        self._plotGame(self.game.get_frame())
  
    def _plotGame(self, frame):
        self.imbuf.seek(0)
        fx, fy = frame.shape[0], frame.shape[1]
        rx, ry = (256, int(fy*256/fx)) if (fx > fy) else (int(fx*256/fy), 256)
        PIL.Image.fromarray((frame*255).astype('uint8')).resize((ry, rx), resample=PIL.Image.NEAREST).save(self.imbuf, 'gif')
        self.canvas.value = self.imbuf.getvalue()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  1 19:49:38 2022

@author: vabi
"""

import chess
import chess.svg
from svglib.svglib import svg2rlg
from reportlab.graphics import renderPM
from cairosvg import svg2png
import imageio
import numpy as np
import math
import io
import PIL.Image as Image
import matplotlib.pyplot as plot

def getPieceDict(): 
    piece_dict = dict()
    piece_dict[None] = 0
    piece_dict['K'] = 1
    piece_dict['Q'] = 2
    piece_dict['R'] = 3
    piece_dict['B'] = 4
    piece_dict['N'] = 5
    piece_dict['P'] = 6
    piece_dict['k'] = 7
    piece_dict['q'] = 8
    piece_dict['r'] = 9
    piece_dict['b'] = 10
    piece_dict['n'] = 11
    piece_dict['p'] = 12
    return piece_dict
    
    

def createChessDiagramData(fen, pieceDict, color):
    board = chess.Board(fen)

    color_dict = dict()
    
    color_dict["square light"] = "#ffffff"
    boardsvg = chess.svg.board(board=board, coordinates=True, orientation=color, colors=color_dict)
    png_data = svg2png(bytestring=boardsvg, output_width=128, output_height=128)

    X = np.array(Image.open(io.BytesIO(png_data)).getdata()).reshape(128, 128, 3)
    Y = np.zeros((8, 8))
    
    Z = np.zeros(1)
    
    if color == chess.BLACK:
        Z[0] = 1
    
    for square in range(64):
        file = square % 8
        rank = math.floor(square/8)
        if board.piece_at(square) != None:
            Y[rank][file] = pieceDict[board.piece_at(square).symbol()]
         
    Y = np.flip(Y, axis=0)
    if color == chess.BLACK:
        Y = np.flip(Y)
        
         
    return X, Y, Z
        
fen = 'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1'
X, Y, Z = createChessDiagramData(fen, getPieceDict(), chess.WHITE)
plot.imshow(X)

#f = open("BoardVisualisedFromFEN.SVG", "w")
#f.write(boardsvg)
#f.close()

#drawing = svg2rlg("BoardVisualisedFromFEN.SVG")
#renderPM.drawToFile(drawing, "file.png", fmt="PNG")



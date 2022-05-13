#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  1 19:49:38 2022

@author: vabi
"""

import os
#os.environ['path'] += r';C:\Program Files\GIMP 2\bin'

#print(os.environ['path'])

import chess
import chess.svg
from svglib.svglib import svg2rlg
from reportlab.graphics import renderPM
from cairosvg import svg2png
import imageio
import numpy as np
import math

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


def getRandomColor(lower, higher):
    r =  int(255*(np.random.random(1)[0]*(higher-lower)+lower))
    g =  int(255*(np.random.random(1)[0]*(higher-lower)+lower))
    b =  int(255*(np.random.random(1)[0]*(higher-lower)+lower))
    return "#"+f'{r:02x}'+f'{g:02x}'+f'{b:02x}'
    
    
    
def get_random_color_dict():
    threshold = 0.33+np.random.random(1)[0]/3
    hysteresis = 0.1
    
    light_threshold = threshold+hysteresis
    dark_threshold  = threshold-hysteresis
    
    colors = dict()
    colors["square light"]      =     getRandomColor(light_threshold, 1)
    colors["coord"]             =     getRandomColor(light_threshold, 1)
    colors["square dark"]       =     getRandomColor(0, dark_threshold)
    colors["margin"]            =     getRandomColor(0, dark_threshold)
    
    return colors


def create_single_data_point(base_path, fen, idx, piece_dict):
  directory = base_path+"diagram"+f'{idx:07d}'
  os.mkdir(directory)
  side = chess.WHITE
  if np.random.rand(1)[0] < 0.5:
      side = chess.BLACK
      
  png_data, Y, Z = create_chess_diagram_data(fen, piece_dict, side, get_random_color_dict(), directory)
  
  np.save(directory+"/piece_data.npy", Y, allow_pickle=False)
  np.save(directory+"/orientation_data.npy", Z, allow_pickle=False)
  
  with open(directory+"/diagram.png", "wb") as f:
      f.write(png_data)
    
def create_data_in_single_directories(num_samples, offset): 
     fens = []
     
     with open('D:\code\ML-playground\chessDiagrams\quiet-labeled.epd', 'r') as f:
         fens = list(f)
         

     piece_dict= getPieceDict()
     
     base_path = "D:\MLData\ChessDiagrams\\"
     
     for ind in range(num_samples):
         if ind % 100 == 0:
             print(ind/num_samples)
         if ind+offset >= len(fens):
             break
         idx = ind+offset

         fen = fens[idx]
         
         parts = fen.split(' ')
         next_fen = parts[0]+' '+parts[1]+' '+parts[2]+' - 0 1'
         create_single_data_point(base_path, next_fen, 2*idx, piece_dict)
         
         rows = parts[0].split('/')
         rows.reverse()
         
         next_fen = ''
         for ind in range(8):
             next_fen = next_fen+rows[ind]
             if ind < 7:
                 next_fen = next_fen+"/"
         next_fen = next_fen+' '+parts[1]+' '+parts[2]+' - 0 1'         
         create_single_data_point(base_path, next_fen, 2*idx+1, piece_dict)             

def create_chess_diagram_data(fen, pieceDict, side, color_dict, directory):
    board = chess.Board(fen)


    
    boardsvg = chess.svg.board(board=board, coordinates=True, orientation=side, colors=color_dict)
    png_data = svg2png(bytestring=boardsvg, output_width=128, output_height=128)
    
    Y = np.zeros((8, 8))
    
    Z = np.zeros(1)
    
    if side == chess.BLACK:
        Z[0] = 1
    
    for square in range(64):
        file = square % 8
        rank = math.floor(square/8)
        if board.piece_at(square) != None:
            Y[rank][file] = pieceDict[board.piece_at(square).symbol()]
         
    Y = np.flip(Y, axis=0)
    if side == chess.BLACK:
        Y = np.flip(Y)
        
         
    return png_data, Y, Z

#fen = 'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1'
#X, Y, Z = createChessDiagramData(fen, getPieceDict())


#f = open("BoardVisualisedFromFEN.SVG", "w")
#f.write(boardsvg)
#f.close()

#drawing = svg2rlg("BoardVisualisedFromFEN.SVG")
#renderPM.drawToFile(drawing, "file.png", fmt="PNG")



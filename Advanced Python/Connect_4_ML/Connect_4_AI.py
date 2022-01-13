#Import all modules needed
import numpy as np
import random
import pygame
import sys
import math
import time

#Define variables that will be used in multiple classes
user = 0
AI = 1
no_fill = 0
user_coin = 1
AI_coin = 2
n_rows = 6
n_columns = 7
n_frame = 4

class Board:

    def create_board():
        board = np.zeros((n_rows,n_columns))
        return board

    def print_board(board):
        print(np.flip(board, 0))

class validation(Board):

    def __init__():
        n_rows = Board.n_rows
        n_columns = Board.n_columns

    def place_coin(board,row, col, coin):

        board[row][col] = coin

    def location_check(board, col):
        return board[n_rows-1][col] == 0

    def get_row(board, col):
        for i in range(n_rows):
            if board[i][col] == 0:
                return i

class winning(validation):

    def win( board, coin):
        # Check horizontal locations for win
        for i in range(n_columns-3):
            for j in range(n_rows):
                if board[j][i] == coin and board[j][i+1] == coin and board[j][i+2] == coin and board[j][i+3] == coin:
                    return True

        # Check vertical locations for win
        for i in range(n_columns):
            for j in range(n_rows-3):
                if board[j][i] == coin and board[j+1][i] == coin and board[j+2][i] == coin and board[j+3][i] == coin:
                    return True

        # Check positively sloped diaganols
        for i in range(n_columns-3):
            for j in range(n_rows-3):
                if board[j][i] == coin and board[j+1][i+1] == coin and board[j+2][i+2] == coin and board[j+3][i+3] == coin:
                    return True

        # Check negatively sloped diaganols
        for i in range(n_columns-3):
            for j in range(3, n_rows):
                if board[j][i] == coin and board[j-1][i+1] == coin and board[j-2][i+2] == coin and board[j-3][i+3] == coin:
                    return True

    def scorer(frame, coin):
        score = 0
        comp_coin = user_coin
        if coin == user_coin:
            comp_coin = AI_coin

        if frame.count(coin) == 4:
            score += 100
        elif frame.count(coin) == 3 and frame.count(no_fill) == 1:
            score += 5
        elif frame.count(coin) == 2 and frame.count(no_fill) == 2:
            score += 2

        if frame.count(comp_coin) == 3 and frame.count(no_fill) == 1:
            score -= 4

        return score

    def score_position(board, coin):
        score = 0

        ## Score middle column
        middle_column = [int(i) for i in list(board[:, n_columns//2])]
        middle_count = middle_column.count(coin)
        score += middle_count * 3

        ## Score Horizontal
        for j in range(n_rows):
            middle_row = [int(i) for i in list(board[j,:])]
            for i in range(n_columns-3):
                frame = middle_row[i:i+n_frame]
                score += winning.scorer(frame, coin)

        ## Score Vertical
        for i in range(n_columns):
            col_array = [int(i) for i in list(board[:,i])]
            for j in range(n_rows-3):
                frame = col_array[j:j+n_frame]
                score += winning.scorer(frame, coin)

        ## Score posiive sloped diagonal
        for j in range(n_rows-3):
            for i in range(n_columns-3):
                frame = [board[j+i][i+i] for i in range(n_frame)]
                score += winning.scorer(frame, coin)

        for j in range(n_rows-3):
            for i in range(n_columns-3):
                frame = [board[j+3-i][i+i] for i in range(n_frame)]
                score += winning.scorer(frame, coin)

        return score

    def good_locations(board):
        good_loc = []
        for col in range(n_columns):
            if validation.location_check(board, col):
                good_loc.append(col)
        return good_loc

    def final_node(board):
        return winning.win(board, user_coin) or winning.win(board, AI_coin) or len(winning.good_locations(board)) == 0


class ai_competitor(winning):

    def ai_motivation(board, depth, alpha, beta, maxfunc):
        good_loc = winning.good_locations(board)
        final = winning.final_node(board)
        if depth == 0 or final:
            if final:
                if winning.win(board, AI_coin):
                    return (None, 100000000000000)
                elif winning.win(board, user_coin):
                    return (None, -10000000000000)
                else: # Game is over, no more valid moves
                    return (None, 0)
            else: # Depth is zero
                return (None, winning.score_position(board, AI_coin))
        if maxfunc:
            value = -math.inf
            column = random.choice(good_loc)
            for col in good_loc:
                row = validation.get_row(board, col)
                b_copy = board.copy()
                validation.place_coin(b_copy,row, col, AI_coin)
                new_score = ai_competitor.ai_motivation(b_copy, depth-1, alpha, beta, False)[1]
                if new_score > value:
                    value = new_score
                    column = col
                alpha = max(alpha, value)
                if alpha >= beta:
                    break
            return column, value

        else: # Minimizing user
            value = math.inf
            column = random.choice(good_loc)
            for col in good_loc:
                row = validation.get_row(board, col)
                b_copy = board.copy()
                validation.place_coin(b_copy, row, col, user_coin)
                new_score = ai_competitor.ai_motivation(b_copy, depth-1, alpha, beta, True)[1]
                if new_score < value:
                    value = new_score
                    column = col
                beta = min(beta, value)
                if alpha >= beta:
                    break
            return column, value

    def best_move(board, coin):

        good_loc = winning.good_locations(board)
        highscore = -10000
        winning_column = random.choice(good_loc)
        for col in good_loc:
            row = validation.get_row(board, col)
            board_copy = board.copy()
            validation.place_coin(board_copy, row, col, coin)
            score = winning.score_position(board_copy, coin)
            if score > highscore:
                highscore = score
                winning_column = col

        return winning_column

class create_game(ai_competitor):

    def create_boardgame(board):
        disp = pygame.display.set_mode((700, 600))
        for i in range(n_columns):
            for j in range(n_rows):
                pygame.draw.rect(disp, (0,0,255), (i*100, j*100+100, 100, 100))
                pygame.draw.circle(disp, (0,0,0), (int(i*100+100/2), int(j*100+100+100/2)), int(100/2 - 5))
        
        for i in range(n_columns):
            for j in range(n_rows):		
                if board[j][i] == user_coin:
                    pygame.draw.circle(disp, (255,0,0), (int(i*100+100/2), 600-int(j*100+100/2)), int(100/2 - 5))
                elif board[j][i] == AI_coin: 
                    pygame.draw.circle(disp, (255,255,0), (int(i*100+100/2), 600-int(j*100+100/2)), int(100/2 - 5))
        pygame.display.update()

    disp = pygame.display.set_mode((700, 600))
    board = Board.create_board()
    Board.print_board(board)
    game_over = False

    pygame.init()

    create_boardgame(board)
    pygame.display.update()

    myfont = pygame.font.SysFont("monospace", 75)

    turn = random.randint(user, AI)

    while not game_over:

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                sys.exit()

            if event.type == pygame.MOUSEMOTION:
                pygame.draw.rect(disp, (0,0,0), (0,0, 700, 100))
                mouse = event.pos[0]
                if turn == user:
                    pygame.draw.circle(disp, (255,0,0), (mouse, int(100/2)), int(100/2 - 5))

            pygame.display.update()

            if event.type == pygame.MOUSEBUTTONDOWN:
                pygame.draw.rect(disp, (0,0,0), (0,0, 700, 100))
                #print(event.pos)
                # Ask for user 1 Input
                if turn == user:
                    mouse = event.pos[0]
                    col = int(math.floor(mouse/100))

                    if validation.location_check(board, col):
                        row = validation.get_row(board, col)
                        validation.place_coin(board, row, col, user_coin)

                        if winning.win(board, user_coin):
                            label = myfont.render("user 1 wins!!", 1, (0,255,0))
                            disp.blit(label, (40,10))
                            time.sleep(5) 
                            game_over = True

                        turn += 1
                        turn = turn % 2

                        Board.print_board(board)
                        create_boardgame(board)


        # # Ask for user 2 Input
        if turn == AI and not game_over:				

            #col = random.randint(0, n_columns-1)
            #col = best_move(board, AI_coin)
            col, ai_motivation_score = ai_competitor.ai_motivation(board, 5, -math.inf, math.inf, True)

            if validation.location_check(board, col):
                #pygame.time.wait(500)
                row = validation.get_row(board, col)
                validation.place_coin(board, row, col, AI_coin)

                if winning.win(board, AI_coin):
                    label = myfont.render("user 2 wins!!", 1, (0,255,0))
                    disp.blit(label, (40,10))
                    time.sleep(5) 
                    game_over = True

                Board.print_board(board)
                create_boardgame(board)

                turn += 1
                turn = turn % 2

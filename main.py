import os
import re
import time
from typing import Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import chess
from google import genai
from google.genai import types
from groq import Groq
from dotenv import load_dotenv

# Load environment variables
load_dotenv(override=True)

app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global game state
current_board = chess.Board()

class AIConfig(BaseModel):
    provider: str  # "gemini" or "groq"
    model: str
    api_key: str

class StepRequest(BaseModel):
    white: AIConfig
    black: AIConfig

class GameStateResponse(BaseModel):
    fen: str
    is_game_over: bool
    result: str
    last_move: str
    thought: str = ""
    error: str = ""

def extract_move_and_thought(text: str) -> tuple[str, str]:
    thought = ""
    move = ""
    
    # Robust parsing for THOUGHT and MOVE
    thought_match = re.search(r"THOUGHT:\s*(.*)", text, re.IGNORECASE)
    if thought_match:
        thought = thought_match.group(1).strip()
        
    move_match = re.search(r"MOVE:\s*([a-h][1-8][a-h][1-8][qrbn]?)", text, re.IGNORECASE)
    if move_match:
        move = move_match.group(1).lower().strip()
    else:
        # Fallback: look for any UCI-like string in the text
        all_moves = re.findall(r"([a-h][1-8][a-h][1-8][qrbn]?)", text, re.IGNORECASE)
        if all_moves:
            move = all_moves[-1].lower().strip()
            
    return thought, move

def get_gemini_move(prompt_text: str, history: str, is_player_white: bool, config: AIConfig) -> tuple[str, str, str]:
    try:
        client = genai.Client(api_key=config.api_key)
        personality = "highly AGGRESSIVE grandmaster" if is_player_white else "tactical and DEFENSIVE grandmaster"
        
        response = client.models.generate_content(
            model=config.model,
            contents=(
                f"You are a {personality} chess engine. Your goal is to WIN. "
                "Briefly explain your strategic reasoning in one sentence, then provide the move in UCI notation (e.g., e2e4). "
                "Format your response exactly as:\n"
                "THOUGHT: [Your one-sentence reasoning]\n"
                "MOVE: [uci_move]\n\n"
                f"Match History: {history}\n\n{prompt_text}"
            ),
            config=types.GenerateContentConfig(temperature=0.7)
        )
        
        if not response or not response.text:
            return "", "", "Empty response from Gemini."
            
        thought, move = extract_move_and_thought(response.text)
        return thought, move, ""
    except Exception as e:
        return "", "", str(e)

def get_groq_move(prompt_text: str, history: str, is_player_white: bool, config: AIConfig) -> tuple[str, str, str]:
    try:
        client = Groq(api_key=config.api_key)
        personality = "highly AGGRESSIVE grandmaster" if is_player_white else "tactical and DEFENSIVE grandmaster"
        
        chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": f"You are a {personality} chess engine. Your goal is to WIN."
                },
                {
                    "role": "user",
                    "content": (
                        "Briefly explain your strategic reasoning in one sentence, then provide the move in UCI notation (e.g., e2e4). "
                        "Format your response exactly as:\n"
                        "THOUGHT: [Your one-sentence reasoning]\n"
                        "MOVE: [uci_move]\n\n"
                        f"Match History: {history}\n\n{prompt_text}"
                    )
                }
            ],
            model=config.model,
            temperature=0.7,
        )
        
        text = chat_completion.choices[0].message.content
        thought, move = extract_move_and_thought(text)
        return thought, move, ""
    except Exception as e:
        return "", "", str(e)

@app.get("/api/state", response_model=GameStateResponse)
def get_state(error: str = ""):
    return GameStateResponse(
        fen=current_board.fen(),
        is_game_over=current_board.is_game_over(),
        result=current_board.result() if current_board.is_game_over() else "*",
        last_move=current_board.peek().uci() if current_board.move_stack else "",
        thought="",
        error=error
    )

@app.post("/api/step", response_model=GameStateResponse)
def play_next_turn(request: StepRequest):
    if current_board.is_game_over():
        return get_state()

    is_white_turn = current_board.turn == chess.WHITE 
    config = request.white if is_white_turn else request.black
    current_engine = f"{config.provider.capitalize()} ({'White' if is_white_turn else 'Black'})"
    
    # Generate legal moves and history
    legal_moves = [move.uci() for move in current_board.legal_moves]
    history_san = chess.Board().variation_san(current_board.move_stack)

    # Identify moves that lead to repetition
    repetition_moves = []
    for move in current_board.legal_moves:
        current_board.push(move)
        if current_board.is_repetition(2):
            repetition_moves.append(move.uci())
        current_board.pop()

    avoid_instruction = ""
    if repetition_moves and len(repetition_moves) < len(legal_moves):
        avoid_instruction = f"\nWARNING: DO NOT play {', '.join(repetition_moves)}. They cause a draw. You MUST choose a different move to win."

    prompt = (
        f"The current board position in FEN format is: {current_board.fen()}\n"
        f"The legal moves you can make are: {', '.join(legal_moves)}\n"
        f"{avoid_instruction}\n"
        f"Make your best move."
    )

    attempts = 0
    chosen_move = None
    final_thought = ""
    last_error = ""

    while attempts < 3:
        if config.provider == "gemini":
            thought, move_str, error = get_gemini_move(prompt, history_san, is_white_turn, config)
        elif config.provider == "groq":
            thought, move_str, error = get_groq_move(prompt, history_san, is_white_turn, config)
        else:
            error = f"Unknown provider: {config.provider}"
            move_str = ""
            thought = ""

        final_thought = thought
        last_error = error
        
        if error:
            attempts += 1
            continue

        try:
            if move_str:
                move = chess.Move.from_uci(move_str)
                if move in current_board.legal_moves:
                    # HARD ENFORCEMENT: Reject repetition moves if others exist
                    current_board.push(move)
                    is_rep = current_board.is_repetition(2)
                    current_board.pop()
                    
                    if is_rep and len(legal_moves) > len(repetition_moves):
                        print(f"Hard-Rejecting {move_str} (repetition) from {current_engine}")
                        attempts += 1
                        continue
                    
                    chosen_move = move
                    break
        except ValueError:
            pass
        attempts += 1

    # Fallback
    if not chosen_move:
        print(f"{current_engine} failed. Fallback engaged. Error: {last_error}")
        safe_moves = [m for m in current_board.legal_moves if m.uci() not in repetition_moves]
        chosen_move = safe_moves[0] if safe_moves else list(current_board.legal_moves)[0]
        final_thought = f"Fallback move played. {f'Error: {last_error}' if last_error else 'AI provided invalid move.'}"

    current_board.push(chosen_move)
    
    state = get_state()
    state.thought = final_thought
    state.error = last_error
    return state

@app.post("/api/reset")
def reset_game():
    global current_board
    current_board = chess.Board()
    return {"status": "reset"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)

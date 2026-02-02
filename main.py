import time
import numpy as np
import torch
import soundfile as sf
from qwen_tts import Qwen3TTSModel
import discord
from discord.ext import commands
import os
import io
import pathlib
from dotenv import load_dotenv

load_dotenv()

TOKEN = os.getenv("DISCORD_TOKEN")

torch.set_float32_matmul_precision('high')

intents = discord.Intents.default()
intents.message_content = True

bot = commands.Bot(command_prefix='!', intents=intents)

model = Qwen3TTSModel.from_pretrained(
    "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
    device_map="cuda:0",
    dtype=torch.bfloat16,
    attn_implementation="sdpa",
)

voice_cache = None

# START OF VOICE CLONE FUNCTIONS
def load_embeddings(folder_path):
    directory = pathlib.Path(folder_path)
    
    loaded_data = {}

    for file_path in directory.glob('*.pt'):
        print(f"Loading {file_path.name}...")
        data = torch.load(file_path, weights_only=False)
        
        loaded_data[file_path.stem] = data

    return loaded_data

voice_cache = load_embeddings("./embedding")

def smooth_append(existing_audio, new_chunk, overlap_samples=200):
    if len(existing_audio) == 0:
        return new_chunk
    
    overlap_samples = min(overlap_samples, len(existing_audio), len(new_chunk))
    
    fade_out = np.linspace(1.0, 0.0, overlap_samples)
    fade_in = np.linspace(0.0, 1.0, overlap_samples)
    
    overlap_zone = (existing_audio[-overlap_samples:] * fade_out) + (new_chunk[:overlap_samples] * fade_in)
    
    return np.concatenate([existing_audio[:-overlap_samples], overlap_zone, new_chunk[overlap_samples:]])

def stream_to_file(text, voice_clone_prompt, file, ctx=None):
    sample_rate = 24000
    full_audio = np.array([], dtype=np.float32) 

    for chunk, chunk_sr in model.stream_generate_voice_clone(
        text=text, 
        language="English", 
        voice_clone_prompt=voice_clone_prompt, 
        emit_every_frames=8, 
        decode_window_frames=80, 
        overlap_samples=512
    ):
        if torch.is_tensor(chunk):
            chunk = chunk.cpu().numpy()
        
        full_audio = smooth_append(full_audio, chunk, overlap_samples=200)
        print(f"Processed chunk. Total length: {len(full_audio)}")

    sf.write(file, full_audio, sample_rate, format="WAV")
    print("file saved")


@bot.event
async def on_ready():
    print(f'Logged in as {bot.user.name}')

@bot.command()
async def join(ctx):
    if ctx.author.voice:
        channel = ctx.author.voice.channel
        
        vc = await channel.connect()
        
        audio_source = 'join.wav'
        
        if os.path.exists(audio_source):
            vc.play(discord.FFmpegPCMAudio(executable="ffmpeg", source=audio_source), 
                    after=lambda e: print(f'finished playing: {e}') if e else None)
        else:
            await ctx.send(f"unable to find '{audio_source}' in my folder.")
    else:
        await ctx.send("user is not currently in a voice channel.")

@bot.command()
async def speak(ctx, voice_name: str, *, prompt: str):
    if ctx.voice_client:
        print("currently in channel")

        vc = ctx.voice_client

        buffer = io.BytesIO()
        stream_to_file(prompt, voice_cache[voice_name], buffer)
        buffer.seek(0)

        source = discord.FFmpegPCMAudio(buffer, pipe=True, executable="ffmpeg")
        
        vc.play(source, after=lambda e: print(f'finished speaking: {e}') if e else None)

    else:
        print("currently not connected to any channels")

@bot.command()
async def leave(ctx):
    if ctx.voice_client:
        await ctx.guild.voice_client.disconnect()
    else:
        await ctx.send("currently disconnected from any voice channel.")

bot.run(TOKEN)
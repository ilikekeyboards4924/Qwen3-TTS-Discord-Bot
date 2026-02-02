# Qwen3-TTS Discord Bot
discord bot i made using Qwen3's new text to speech model, specifically a fork of dffdeeq's version of the model that has real-time streaming,
something that the Qwen3 tts research paper claimed that it had, but it DOESN'T have it. anyways that you dffdeeq for the real-time streaming, much appreciated.

if you want to add your own voice to the bot, just create a `voice_clone_prompt` with the aforementioned model and save the resulting `.pt` file in the embedding folder.
make sure to use the `1.7b` parameter model when generating the `voice_clone_prompt` and not the `0.6b` parameter model.

make sure you enable all the correct permissions for the bot in the discord developer portal as well.
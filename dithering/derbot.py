import discord
from discord.ext import commands
import discord.utils
import dithering
import numpy as np
import cv2
import os
import math
from dotenv import load_dotenv

load_dotenv()

intents = discord.Intents().all()
bot = commands.Bot(command_prefix="käs", intents=intents)

# eifach en screenshot gmacht und ungefähr gmesse - hoffe es isch konstant bi allne grät
h_by_w = 662/779

@bot.command(brief='make de dithering', description='dither bither (i ha kein plan man)')
async def ig(ctx):
    try:
        for file in ctx.message.attachments:
            image_bytes = await file.read()

            np_array = np.frombuffer(image_bytes, np.uint8)
            cv_image = cv2.imdecode(np_array, cv2.IMREAD_COLOR)
            (h, w, c) = cv_image.shape
            cv_image = cv2.resize(cv_image, (60, round(60 * h / w * h_by_w)))
            print(cv_image.shape)

            await ctx.send("der käs kommt bald...")

            available_emojis = [((0, ":white_large_square:"), (255, 255, 255))]
            for emoji in ctx.guild.emojis:
                image_bytes = await emoji.read()

                np_array = np.frombuffer(image_bytes, np.uint8)
                cv_emoji = cv2.imdecode(np_array, cv2.IMREAD_COLOR)

                available_emojis.append(((emoji.id, emoji.name), dithering.get_average_color(cv_emoji)))
                print(emoji.name, dithering.get_average_color(cv_emoji))

            print(available_emojis)

            print(cv_image[0][0])
            print(dithering.closestColor(cv_image[0][0], available_emojis))
            dithered = dithering.floydSteinberg(cv_image, available_emojis)
            #dithered = dithering.longusMongus(cv_image, available_emojis)
            for line in dithered:
                msg = ""
                for emoji_id, emoji_name in line:
                    if emoji_id != 0:
                        msg += str(bot.get_emoji(emoji_id))
                    else:
                        msg += emoji_name
                if len(msg) > 2000:
                    await ctx.send("Fehler: Nachricht zu lang: {} Zeichen".format(len(msg)))
                    return
                await ctx.send(msg)

            await ctx.send("der käs ist fertig...")
    except Exception as e:
        await ctx.send(str(e))



# token & groups for commands
#bot.add_cog(ownerOnly(bot))
#bot.add_cog(ingameOnly(bot))
bot.run(os.getenv("MYKEY"))

print(bot.emojis)

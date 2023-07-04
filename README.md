# DiscordDocChatBot 🤖

Welcome to the DiscordDocChatBot project! This bot uses the OpenAI API to engage in document-oriented conversations within a Discord server.

Please note that this bot is still under development and not yet ready for production. Also, it's not affiliated with OpenAI or Discord.

## 🛠️ Installation 

Before you start, ensure you have Python and Docker installed on your machine. 

### Python 

If you don't have Python installed, you can download it from the [official website](https://www.python.org/downloads/). Select the version suitable for your operating system and follow the provided instructions.

### Docker 

Docker can be downloaded from the [Docker official website](https://www.docker.com/products/docker-desktop). Choose the appropriate version for your OS. Follow the installation instructions provided on the website.

## 📚 Prerequisites

To successfully run this bot, you'll need to set up the following environment variables:

1. **OpenAI API Token:** Visit [OpenAI Platform](https://platform.openai.com/account/api-keys) to obtain your API key.

    ```
    OPENAI_API_KEY=''
    ```

2. **Discord Bot Token:** You need to create a Discord bot and add it to your server. Check the official [Discord Developers Documentation](https://discord.com/developers/docs/intro) for a step-by-step guide.

    ```
    DISCORD_TOKEN=''
    ```

3. **HuggingFace API Token:** You can find your token at [HuggingFace](https://huggingface.co/settings/token).

    ```
    HUGGINGFACE_API_TOKEN=''
    HUGGINGFACEHUB_API_TOKEN=''
    ```

4. **Google API Token:** Create a project and enable the YouTube Data API V3 at [Google Cloud Console](https://console.cloud.google.com/apis/credentials?project=angular-unison-352106). This token is used for the Google search command.

    ```
    GOOGLE_API_KEY=''
    ```

## 🚀 Deployment

Once the environment variables are in place, execute the following Docker commands to run the bot:

```shell
docker build -t discorddocchatbot .
docker-compose up -d

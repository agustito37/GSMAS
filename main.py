from openai import OpenAI

import config

client = OpenAI(api_key=config.OPENAI_API_KEY)


def main() -> None:
    resp = client.responses.create(
        model="gpt-4.1-mini",
        input="Say hello in one short sentence.",
    )
    print(resp.output_text)


if __name__ == "__main__":
    main()

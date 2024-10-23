from typing import Optional

import fire
import sys

from models.llama3.api.datatypes import (
    CompletionMessage,
    StopReason,
    SystemMessage,
    UserMessage,
)

from models.llama3.reference_impl.generation import Llama


def run_main(
    ckpt_dir: str,
    temperature: float = 0.6,
    top_p: float = 0.9,
    max_seq_len: int = 512,
    max_batch_size: int = 4,
    max_gen_len: Optional[int] = None,
    model_parallel_size: Optional[int] = None,
):
    """
    Interactive chat loop with the Llama model.
    """
    generator = Llama.build(
        ckpt_dir=ckpt_dir,
        max_seq_len=max_seq_len,
        max_batch_size=max_batch_size,
        model_parallel_size=model_parallel_size,
    )

    # Initialize conversation history
    conversation = []

    print("Welcome to the Llama Chat! Type 'exit' or 'quit' to end the chat.\n")

    while True:
        try:
            user_input = input("User: ").strip()
            if user_input.lower() in {"exit", "quit"}:
                print("Exiting chat. Goodbye!")
                break

            # Add user message to the conversation
            conversation.append(UserMessage(content=user_input))

            # Generate response
            result = generator.chat_completion(
                conversation,
                max_gen_len=max_gen_len,
                temperature=temperature,
                top_p=top_p,
            )

            # Get the generated message
            out_message = result.generation

            # Add assistant's response to the conversation
            conversation.append(out_message)

            # Display the assistant's response
            print(f"{out_message.role.capitalize()}: {out_message.content}\n")

            # Optional: Limit conversation history to prevent it from growing indefinitely
            if len(conversation) > 20:
                # Remove the oldest two messages (user and assistant)
                conversation = conversation[-18:]

        except KeyboardInterrupt:
            print("\nInterrupted by user. Exiting chat.")
            sys.exit(0)
        except Exception as e:
            print(f"An error occurred: {e}")
            continue


def main():
    fire.Fire(run_main)


if __name__ == "__main__":
    main()

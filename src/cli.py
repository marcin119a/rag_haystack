import sys
import uuid
from agents.workflow import workflow

def main() -> None:
    if len(sys.argv) > 1:
        pytanie = " ".join(sys.argv[1:])
        print(workflow.run(pytanie))
        return

    print("Doradca szkoleniowy (wpisz 'exit', aby zakończyć)")
    session_id = str(uuid.uuid4())
    while True:
        try:
            pytanie = input("\nTy: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not pytanie:
            continue
        if pytanie.lower() in {"exit", "quit"}:
            break
        print()
        print(workflow.run(pytanie).content)


if __name__ == "__main__":
    main()

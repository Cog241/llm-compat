'''Allow ``python -m llm_compat`` to behave like the console script.'''

from llm_compat.cli import main

if __name__ == '__main__':
    raise SystemExit(main())

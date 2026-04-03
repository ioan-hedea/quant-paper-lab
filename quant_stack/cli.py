"""Console-script entrypoints for the installable research package."""

from __future__ import annotations


def pipeline_main() -> None:
    from .main import main

    main()


def research_main() -> None:
    from quant_research import main

    main()


def plots_main() -> None:
    from scripts.generate_control_story_plots import main

    main()

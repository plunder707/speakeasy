#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
tui_prototype.py - Phase 0 TUI Prototype
Minimal proof-of-concept for Resonance Workspace TUI.

Tests:
- rich.Live() rendering without flicker
- sidecar.py async/sync streaming
- Real-time token display
- Status bar updates
"""

import sys
from typing import Optional
from dataclasses import dataclass
from rich.console import Console
from rich.layout import Layout
from rich.panel import Panel
from rich.live import Live
from rich.text import Text
from rich.table import Table

# Import sidecar for async/sync bridging
try:
    from sidecar import Sidecar
except ImportError:
    print("❌ Critical: 'sidecar.py' not found.")
    sys.exit(1)

# Import backend components
try:
    from wiki_governor import (
        VectorKnowledgeBase,
        fetch_wikipedia_full,
        fetch_arxiv_full,
        fetch_web_full,
        understand_images_with_llava,
        get_search_info
    )
    from resonance_governor import OllamaResonanceWrapper
except ImportError as e:
    print(f"❌ Critical: Backend import failed: {e}")
    sys.exit(1)


@dataclass
class UIState:
    """Thread-safe UI state"""
    verified_text: str = ""
    current_drift: float = 0.0
    source_count: int = 0
    token_count: int = 0
    status: str = "Ready"


class ResonanceTUI:
    """Minimal TUI prototype using rich + sidecar"""

    def __init__(self):
        self.console = Console()
        self.bridge = Sidecar(workers=2, name="TUI-Bridge")
        self.state = UIState()

    def build_layout(self) -> Layout:
        """Build the UI layout"""
        layout = Layout()

        # Split into main area + status bar
        layout.split_column(
            Layout(name="main", ratio=9),
            Layout(name="status", size=3)
        )

        # Main content panel
        content = Panel(
            Text(self.state.verified_text, style="white"),
            title="[cyan]Resonance Governor Output[/cyan]",
            border_style="cyan"
        )
        layout["main"].update(content)

        # Status bar
        status_table = Table.grid(expand=True)
        status_table.add_column(justify="left")
        status_table.add_column(justify="center")
        status_table.add_column(justify="right")

        drift_color = "green" if self.state.current_drift < 0.5 else "yellow" if self.state.current_drift < 0.85 else "red"

        status_table.add_row(
            f"[white]Status:[/white] {self.state.status}",
            f"[{drift_color}]Drift: {self.state.current_drift:.3f}[/{drift_color}]",
            f"[white]Sources: {self.state.source_count} | Tokens: {self.state.token_count}[/white]"
        )

        status_panel = Panel(status_table, style="blue")
        layout["status"].update(status_panel)

        return layout

    def run_query_sync(self, query: str):
        """
        Run a query using the existing synchronous backend.
        This is the Phase 0 approach - just display results as they come.
        """
        self.state.status = "Searching..."
        self.state.verified_text = ""
        self.state.token_count = 0

        # Search phase
        info = get_search_info(query)
        self.state.status = "Fetching documents..."

        # Fetch documents (synchronous for now)
        all_documents = []
        all_image_paths = []
        session_id = "tui_session_1"

        # Wikipedia
        all_documents.extend(fetch_wikipedia_full(info['web_query']))

        # arXiv
        if info['arxiv_search']:
            all_documents.extend(fetch_arxiv_full(info['arxiv_query']))

        # Web scraping
        web_docs, web_images = fetch_web_full(info['web_query'], session_id)
        all_documents.extend(web_docs)
        all_image_paths.extend(web_images)

        # Image understanding
        if all_image_paths:
            unique_image_paths = list(dict.fromkeys(all_image_paths))
            image_docs = understand_images_with_llava(unique_image_paths)
            all_documents.extend(image_docs)

        self.state.source_count = len(all_documents)

        if not all_documents:
            self.state.status = "No documents found"
            return

        # Vector DB
        self.state.status = "Building vector index..."
        kb = VectorKnowledgeBase()
        kb.add_documents(all_documents, session_id)

        # Retrieval
        self.state.status = "Retrieving context..."
        retrieved_context = kb.retrieve(query, session_id, top_k=30)

        if len(retrieved_context) < 100:
            self.state.status = "Insufficient context"
            return

        # Generation phase
        self.state.status = "Generating response..."

        try:
            wrapper = OllamaResonanceWrapper(
                retrieved_context,
                model="llama3",
                enable_diagrams=True
            )

            # Stream tokens
            for chunk in wrapper.stream_chat(query):
                self.state.verified_text += chunk
                self.state.token_count += len(chunk.split())
                # Note: drift would need to be exposed from wrapper for live updates

            self.state.status = "Complete"

        except Exception as e:
            self.state.status = f"Error: {e}"

    def run(self, query: str):
        """Run the TUI with a query"""

        with Live(self.build_layout(), refresh_per_second=10, console=self.console) as live:
            # Update UI as we process
            try:
                self.run_query_sync(query)

                # Keep updating UI to show final state
                while self.state.status != "Complete":
                    live.update(self.build_layout())

                # Final update
                live.update(self.build_layout())

            except KeyboardInterrupt:
                self.state.status = "Interrupted"
                live.update(self.build_layout())
            except Exception as e:
                self.state.status = f"Fatal: {e}"
                live.update(self.build_layout())

        # Show final output after TUI closes
        self.console.print(f"\n[green]✓ Complete[/green]\n")


def main():
    if len(sys.argv) < 2:
        print("Usage: python3 tui_prototype.py '<query>'")
        print("Example: python3 tui_prototype.py 'Explain Calvin cycle pathway'")
        sys.exit(1)

    query = " ".join(sys.argv[1:])

    tui = ResonanceTUI()
    tui.run(query)


if __name__ == "__main__":
    main()

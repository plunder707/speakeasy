#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
resonance_tui.py - Resonance Workspace TUI (Phase 0)

Bloomberg Terminal aesthetic for AI Governor visualization.
Real-time streaming with source tracking and status updates.

Architecture:
- Left Pane: Source list (populated as fetched)
- Right Pane: Streaming output with status bar
- Powered by: sidecar.py (async/sync) + rich.Live() (rendering)
"""

import sys
import asyncio
from pathlib import Path
from typing import List, Optional
from dataclasses import dataclass, field

from rich.console import Console
from rich.layout import Layout
from rich.panel import Panel
from rich.live import Live
from rich.table import Table
from rich.text import Text
from rich.progress import Progress, SpinnerColumn, TextColumn

# Sidecar for async/sync bridging
try:
    from sidecar import Sidecar
except ImportError:
    print("❌ Critical: 'sidecar.py' not found")
    sys.exit(1)

# Backend components
try:
    from wiki_governor import (
        VectorKnowledgeBase,
        fetch_wikipedia_full,
        fetch_arxiv_full,
        fetch_web_full,
        understand_images_with_llava,
        get_search_info,
        Document
    )
    from resonance_governor import OllamaResonanceWrapper
except ImportError as e:
    print(f"❌ Backend import failed: {e}")
    sys.exit(1)


@dataclass
class Source:
    """Represents a retrieved source"""
    title: str
    doc_type: str  # wikipedia, arxiv, web, image
    char_count: int
    status: str = "fetched"  # fetched, processing, indexed


@dataclass
class TUIState:
    """Global TUI state - thread-safe via sidecar"""
    # Phase tracking
    phase: str = "Initializing"
    phase_detail: str = ""

    # Sources
    sources: List[Source] = field(default_factory=list)
    total_sources: int = 0
    total_chars: int = 0

    # Output
    output_text: str = ""
    token_count: int = 0

    # Metrics
    drift_score: float = 0.0
    chunks_retrieved: int = 0

    # Flags
    complete: bool = False
    error: Optional[str] = None


class ResonanceTUI:
    """
    Professional TUI for Resonance Governor
    Phase 0: 2-pane layout with real-time updates
    """

    def __init__(self):
        self.console = Console()
        self.bridge = Sidecar(workers=2, name="ResonanceTUI")
        self.state = TUIState()
        self.session_id = "tui_session"

    def build_layout(self) -> Layout:
        """Build the 2-pane layout"""
        layout = Layout()

        # Main split: Sources (30%) | Output (70%)
        layout.split_row(
            Layout(name="sources", ratio=3),
            Layout(name="main", ratio=7)
        )

        # Left pane: Source list
        layout["sources"].update(self._build_source_panel())

        # Right pane: Split into output + status
        layout["main"].split_column(
            Layout(name="output", ratio=9),
            Layout(name="status", size=3)
        )

        layout["main"]["output"].update(self._build_output_panel())
        layout["main"]["status"].update(self._build_status_bar())

        return layout

    def _build_source_panel(self) -> Panel:
        """Build the source tracking panel (left pane)"""
        if not self.state.sources:
            content = Text("Awaiting sources...", style="dim")
        else:
            # Group by type
            wiki_sources = [s for s in self.state.sources if s.doc_type == "wikipedia"]
            web_sources = [s for s in self.state.sources if s.doc_type == "web"]
            image_sources = [s for s in self.state.sources if s.doc_type == "image"]

            lines = []

            if wiki_sources:
                lines.append(Text("📚 Wikipedia", style="bold cyan"))
                for src in wiki_sources[:3]:  # Show top 3
                    lines.append(Text(f"  • {src.title[:30]}...", style="white"))
                if len(wiki_sources) > 3:
                    lines.append(Text(f"  [+{len(wiki_sources)-3} more]", style="dim"))
                lines.append(Text(""))

            if web_sources:
                lines.append(Text("🌐 Web Sources", style="bold cyan"))
                for src in web_sources[:3]:
                    lines.append(Text(f"  • {src.title[:30]}...", style="white"))
                if len(web_sources) > 3:
                    lines.append(Text(f"  [+{len(web_sources)-3} more]", style="dim"))
                lines.append(Text(""))

            if image_sources:
                lines.append(Text("🖼️  Images", style="bold cyan"))
                lines.append(Text(f"  • {len(image_sources)} analyzed", style="white"))
                lines.append(Text(""))

            # Summary
            lines.append(Text("─" * 30, style="dim"))
            lines.append(Text(f"Total: {self.state.total_sources} sources", style="bold white"))
            lines.append(Text(f"Data: {self.state.total_chars:,} chars", style="white"))

            content = Text("\n").join(lines)

        return Panel(
            content,
            title="[yellow]⚓ Anchor Sources[/yellow]",
            border_style="yellow",
            padding=(1, 2)
        )

    def _build_output_panel(self) -> Panel:
        """Build the main output panel"""
        if self.state.error:
            content = Text(self.state.error, style="bold red")
        elif not self.state.output_text:
            content = Text("Preparing response...", style="dim italic")
        else:
            content = Text(self.state.output_text, style="cyan")

        title_style = "green" if self.state.complete else "cyan"
        title = f"[{title_style}]🎯 Resonance Output[/{title_style}]"

        return Panel(
            content,
            title=title,
            border_style=title_style,
            padding=(1, 2)
        )

    def _build_status_bar(self) -> Panel:
        """Build the bottom status bar"""
        status_table = Table.grid(expand=True)
        status_table.add_column(justify="left", ratio=3)
        status_table.add_column(justify="center", ratio=2)
        status_table.add_column(justify="right", ratio=3)

        # Left: Phase
        phase_text = f"[white]{self.state.phase}[/white]"
        if self.state.phase_detail:
            phase_text += f" [dim]{self.state.phase_detail}[/dim]"

        # Center: Drift (prominent display with visual indicator)
        drift_color = "green" if self.state.drift_score < 0.5 else "yellow" if self.state.drift_score < 0.85 else "red"
        drift_icon = "✓" if self.state.drift_score < 0.5 else "⚠" if self.state.drift_score < 0.85 else "✗"
        drift_text = f"[bold {drift_color}]{drift_icon} Drift: {self.state.drift_score:.3f}[/bold {drift_color}]"

        # Right: Metrics
        metrics_text = f"[white]Tokens: {self.state.token_count}[/white]"
        if self.state.chunks_retrieved > 0:
            metrics_text += f" [dim]| Chunks: {self.state.chunks_retrieved}[/dim]"

        status_table.add_row(phase_text, drift_text, metrics_text)

        return Panel(status_table, style="blue")

    async def async_run_query(self, query: str):
        """
        Async version of query execution
        This allows UI updates during processing
        """
        try:
            # Phase 1: Search
            self.state.phase = "Analyzing Query"
            await asyncio.sleep(0.1)  # Allow UI update

            info = get_search_info(query)

            # Phase 2: Fetch documents
            self.state.phase = "Fetching Sources"
            self.state.phase_detail = "Wikipedia..."

            all_documents = []
            all_image_paths = []

            # Wikipedia
            wiki_docs = fetch_wikipedia_full(info['web_query'])
            all_documents.extend(wiki_docs)
            for doc in wiki_docs:
                self.state.sources.append(Source(doc.title, "wikipedia", len(doc.content)))
                self.state.total_sources += 1
                self.state.total_chars += len(doc.content)
            await asyncio.sleep(0.1)

            # arXiv
            if info['arxiv_search']:
                self.state.phase_detail = "arXiv papers..."
                arxiv_docs = fetch_arxiv_full(info['arxiv_query'])
                all_documents.extend(arxiv_docs)
                for doc in arxiv_docs:
                    self.state.sources.append(Source(doc.title, "arxiv", len(doc.content)))
                    self.state.total_sources += 1
                    self.state.total_chars += len(doc.content)
                await asyncio.sleep(0.1)

            # Web scraping
            self.state.phase_detail = "Web sources..."
            web_docs, web_images = fetch_web_full(info['web_query'], self.session_id)
            all_documents.extend(web_docs)
            all_image_paths.extend(web_images)
            for doc in web_docs:
                self.state.sources.append(Source(doc.title, "web", len(doc.content)))
                self.state.total_sources += 1
                self.state.total_chars += len(doc.content)
            await asyncio.sleep(0.1)

            # Image understanding
            if all_image_paths:
                self.state.phase = "Analyzing Images"
                unique_images = list(dict.fromkeys(all_image_paths))
                self.state.phase_detail = f"{len(unique_images)} images with LLaVA..."

                image_docs = understand_images_with_llava(unique_images)
                all_documents.extend(image_docs)
                for doc in image_docs:
                    self.state.sources.append(Source(doc.title[:40], "image", len(doc.content)))
                    self.state.total_sources += 1
                    self.state.total_chars += len(doc.content)
                await asyncio.sleep(0.1)

            if not all_documents:
                self.state.error = "No sources retrieved"
                return

            # Phase 3: Vector DB
            self.state.phase = "Building Knowledge Base"
            self.state.phase_detail = f"{len(all_documents)} documents..."

            kb = VectorKnowledgeBase()
            kb.add_documents(all_documents, self.session_id)
            await asyncio.sleep(0.1)

            # Phase 4: Retrieval
            self.state.phase = "Retrieving Context"
            retrieved_context = kb.retrieve(query, self.session_id, top_k=30)
            self.state.chunks_retrieved = 30

            if len(retrieved_context) < 100:
                self.state.error = "Insufficient context retrieved"
                return

            # Phase 5: Generation
            self.state.phase = "Generating Response"
            self.state.phase_detail = "Streaming from Llama3..."

            wrapper = OllamaResonanceWrapper(
                retrieved_context,
                model="llama3",
                enable_diagrams=True
            )

            # Stream tokens and update drift score
            for chunk in wrapper.stream_chat(query):
                self.state.output_text += chunk
                self.state.token_count += len(chunk.split())
                self.state.drift_score = wrapper.current_drift  # Live drift tracking
                await asyncio.sleep(0)  # Yield control for UI updates

            self.state.phase = "Complete"
            self.state.phase_detail = ""
            self.state.complete = True

        except Exception as e:
            self.state.error = f"Error: {e}"
            self.state.phase = "Failed"

    def run(self, query: str):
        """Run the TUI with live updates"""
        with Live(
            self.build_layout(),
            refresh_per_second=4,  # Reduced to minimize flashing
            console=self.console,
            screen=False  # Don't use alt screen - keeps output visible
        ) as live:
            # Create async task
            async def update_loop():
                # Run query in background
                query_task = asyncio.create_task(self.async_run_query(query))

                # Update UI while query runs
                while not self.state.complete and not self.state.error:
                    live.update(self.build_layout())
                    await asyncio.sleep(0.25)  # 4fps

                # Wait for query to finish
                await query_task

                # Final update - keep showing for user to read
                live.update(self.build_layout())

            # Run the async loop
            try:
                asyncio.run(update_loop())
            except KeyboardInterrupt:
                self.state.error = "Interrupted by user"
                self.state.phase = "Cancelled"
                live.update(self.build_layout())

        # After exiting Live context, show summary
        self.console.print("\n" + "="*60)
        if self.state.complete:
            self.console.print("[green]✓ Session Complete[/green]")
            self.console.print(f"  📚 Sources: {self.state.total_sources} documents ({self.state.total_chars:,} chars)")
            self.console.print(f"  💬 Tokens Generated: {self.state.token_count}")

            drift_color = 'green' if self.state.drift_score < 0.5 else 'yellow' if self.state.drift_score < 0.85 else 'red'
            drift_emoji = '✓' if self.state.drift_score < 0.5 else '⚠' if self.state.drift_score < 0.85 else '✗'
            self.console.print(f"  {drift_emoji} Final Drift: [{drift_color}]{self.state.drift_score:.3f}[/] ({'LOW - Good!' if self.state.drift_score < 0.5 else 'MEDIUM - Watch for hallucination' if self.state.drift_score < 0.85 else 'HIGH - Likely hallucinating'})")
        elif self.state.error:
            self.console.print(f"[red]✗ Error: {self.state.error}[/red]")

        self.console.print("="*60 + "\n")

        # Cleanup
        self.bridge.shutdown()


def main():
    if len(sys.argv) < 2:
        print("Usage: python3 resonance_tui.py '<query>'")
        print("Example: python3 resonance_tui.py 'Explain the Calvin cycle'")
        sys.exit(1)

    query = " ".join(sys.argv[1:])

    tui = ResonanceTUI()
    tui.run(query)


if __name__ == "__main__":
    main()

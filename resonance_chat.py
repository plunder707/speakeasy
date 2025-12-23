#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
resonance_chat.py - Persistent Chat Interface for Resonance Governor

Professional TUI with:
- Persistent conversation (multiple questions)
- Scrollable history with copy/paste support
- Real-time source tracking
- Live drift monitoring
- Input box always visible at bottom
"""

# Suppress TensorFlow warnings BEFORE any imports
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow warnings
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Disable oneDNN warnings

import asyncio
import sys
import contextlib
from datetime import datetime
from textual.app import App, ComposeResult
from textual.containers import Container, Horizontal, Vertical, ScrollableContainer
from textual.widgets import Header, Footer, Input, Static, RichLog
from textual.binding import Binding
from rich.text import Text
from rich.panel import Panel
from rich.table import Table

# Import backend components
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


@contextlib.contextmanager
def suppress_output():
    """
    Context manager to suppress Python-level output (stdout + stderr).
    Safe for use with Textual - doesn't interfere with terminal handling.
    Note: Won't catch C library output (TensorFlow warnings will still appear)
    """
    import io

    # Save original streams
    old_stdout = sys.stdout
    old_stderr = sys.stderr

    try:
        # Redirect to StringIO (Python level only)
        sys.stdout = io.StringIO()
        sys.stderr = io.StringIO()

        yield

    finally:
        # Restore original streams
        sys.stdout = old_stdout
        sys.stderr = old_stderr


class SourcePanel(Static):
    """Sidebar showing source tracking and metrics"""

    def __init__(self):
        super().__init__()
        self.sources_wiki = 0
        self.sources_web = 0
        self.sources_images = 0
        self.total_chars = 0
        self.drift = 0.0
        self.tokens = 0
        self.chunks = 0
        self.status = "Ready"

    def render(self) -> Panel:
        """Render the source panel"""
        table = Table.grid(padding=(0, 1))
        table.add_column(justify="left", style="cyan")

        # Status
        table.add_row(f"[bold]STATUS:[/bold] {self.status}")
        table.add_row("")

        # Sources
        table.add_row("[bold]SOURCES:[/bold]")
        table.add_row(f"  Wikipedia: {self.sources_wiki}")
        table.add_row(f"  Web: {self.sources_web}")
        table.add_row(f"  Images: {self.sources_images}")
        table.add_row(f"  Total Data: {self.total_chars:,} chars")
        table.add_row("")

        # Metrics
        table.add_row("[bold]METRICS:[/bold]")

        # Drift with color coding
        drift_color = "green" if self.drift < 0.5 else "yellow" if self.drift < 0.85 else "red"
        drift_status = "LOW" if self.drift < 0.5 else "MED" if self.drift < 0.85 else "HIGH"
        table.add_row(f"  Drift: [{drift_color}]{self.drift:.3f}[/{drift_color}] ({drift_status})")

        table.add_row(f"  Tokens: {self.tokens}")
        table.add_row(f"  Chunks: {self.chunks}")

        return Panel(table, title="[cyan]Anchor Sources[/cyan]", border_style="cyan")

    def update_status(self, status: str):
        self.status = status
        self.refresh()

    def update_sources(self, wiki=0, web=0, images=0, chars=0):
        self.sources_wiki += wiki
        self.sources_web += web
        self.sources_images += images
        self.total_chars += chars
        self.refresh()

    def update_metrics(self, drift=None, tokens=None, chunks=None):
        if drift is not None:
            self.drift = drift
        if tokens is not None:
            self.tokens = tokens
        if chunks is not None:
            self.chunks = chunks
        self.refresh()

    def reset_session(self):
        """Reset for new query"""
        self.sources_wiki = 0
        self.sources_web = 0
        self.sources_images = 0
        self.total_chars = 0
        self.drift = 0.0
        self.tokens = 0
        self.chunks = 0
        self.refresh()


class ResonanceChat(App):
    """Persistent chat interface for Resonance Governor"""

    CSS = """
    #main-container {
        layout: horizontal;
        height: 100%;
    }

    #sidebar {
        width: 35;
        dock: left;
    }

    #chat-container {
        width: 1fr;
        layout: vertical;
    }

    #conversation {
        height: 1fr;
        border: solid cyan;
        overflow-y: auto;
        padding: 1;
    }

    #input-box {
        height: 3;
        dock: bottom;
        border: solid green;
    }

    Input {
        border: none;
    }

    Input:disabled {
        opacity: 0.5;
    }
    """

    BINDINGS = [
        Binding("ctrl+q", "quit", "Quit", show=True),
        Binding("ctrl+s", "save_conversation", "Save", show=True),
        Binding("ctrl+e", "export_last", "Export", show=True),
        Binding("ctrl+l", "clear", "Clear", show=False),
    ]

    def __init__(self):
        super().__init__()
        self.kb = VectorKnowledgeBase()
        self.session_counter = 0
        self.current_wrapper = None
        self.conversation_history = []  # Track full conversation
        self.last_assistant_response = ""  # Track last response for export

    def compose(self) -> ComposeResult:
        """Create child widgets"""
        yield Header(show_clock=True)

        with Container(id="main-container"):
            # Sidebar
            with Vertical(id="sidebar"):
                yield SourcePanel()

            # Main chat area
            with Vertical(id="chat-container"):
                yield RichLog(
                    id="conversation",
                    highlight=True,
                    markup=True,
                    wrap=True  # Enable text wrapping (no horizontal scroll)
                )
                yield Input(
                    placeholder="Ask a question (Ctrl+Q to quit)...",
                    id="input-box"
                )

        yield Footer()

    def on_mount(self) -> None:
        """Initialize the app"""
        conversation = self.query_one("#conversation", RichLog)
        conversation.write("[bold cyan]Resonance Governor - Interactive Chat[/bold cyan]")
        conversation.write("[dim]Powered by: Wikipedia | arXiv | Web | LLaVA Vision | LanceDB Vector Search[/dim]")
        conversation.write("")
        conversation.write("[green]Ready. Ask your first question below.[/green]")

        # Focus the input box
        self.query_one("#input-box", Input).focus()

    async def on_input_submitted(self, event: Input.Submitted) -> None:
        """Handle user input submission"""
        query = event.value.strip()

        if not query:
            return

        # Check for quit command
        if query.lower() in ['quit', 'exit', 'q']:
            self.exit()
            return

        # Get input widget
        input_box = self.query_one("#input-box", Input)

        # Clear input and disable during processing
        input_box.value = ""
        input_box.disabled = True
        input_box.placeholder = "Processing... please wait"

        # Log user query
        conversation = self.query_one("#conversation", RichLog)
        conversation.write("")
        conversation.write(f"[bold]You:[/bold] {query}")
        conversation.write("")

        # Track in history
        self.conversation_history.append(f"\n[USER] {query}\n")

        # Process query asynchronously
        await self.process_query(query)

        # Re-enable input after processing
        input_box.disabled = False
        input_box.placeholder = "Ask a question (Ctrl+Q to quit)..."
        input_box.focus()

    async def process_query(self, query: str):
        """Process a user query through the full RAG pipeline"""
        conversation = self.query_one("#conversation", RichLog)
        source_panel = self.query_one(SourcePanel)

        # Increment session
        self.session_counter += 1
        session_id = f"session_{self.session_counter}"

        # Reset source panel for new query
        source_panel.reset_session()
        source_panel.update_status("Analyzing query...")

        try:
            # Step 1: Analyze query
            info = get_search_info(query)

            # Step 2: Fetch documents
            source_panel.update_status("Fetching sources...")
            all_documents = []
            all_image_paths = []

            # Wikipedia
            source_panel.update_status("Fetching Wikipedia...")
            conversation.write("[dim]Searching Wikipedia...[/dim]")

            with suppress_output():
                wiki_docs = await asyncio.to_thread(fetch_wikipedia_full, info['web_query'])

            all_documents.extend(wiki_docs)
            source_panel.update_sources(
                wiki=len(wiki_docs),
                chars=sum(len(d.content) for d in wiki_docs)
            )
            conversation.write(f"[dim]Found {len(wiki_docs)} Wikipedia articles[/dim]")

            # arXiv
            if info['arxiv_search']:
                source_panel.update_status("Fetching arXiv...")
                conversation.write("[dim]Searching arXiv papers...[/dim]")

                with suppress_output():
                    arxiv_docs = await asyncio.to_thread(fetch_arxiv_full, info['arxiv_query'])

                all_documents.extend(arxiv_docs)
                source_panel.update_sources(chars=sum(len(d.content) for d in arxiv_docs))
                conversation.write(f"[dim]Found {len(arxiv_docs)} arXiv papers[/dim]")

            # Web scraping
            source_panel.update_status("Scraping web...")
            conversation.write("[dim]Scraping web sources...[/dim]")

            with suppress_output():
                web_docs, web_images = await asyncio.to_thread(
                    fetch_web_full, info['web_query'], session_id
                )

            all_documents.extend(web_docs)
            all_image_paths.extend(web_images)
            source_panel.update_sources(
                web=len(web_docs),
                chars=sum(len(d.content) for d in web_docs)
            )
            conversation.write(f"[dim]Scraped {len(web_docs)} web pages, found {len(web_images)} images[/dim]")

            # Image understanding
            if all_image_paths:
                unique_images = list(dict.fromkeys(all_image_paths))
                source_panel.update_status(f"Analyzing {len(unique_images)} images...")
                conversation.write(f"[dim]Analyzing {len(unique_images)} images with LLaVA...[/dim]")

                with suppress_output():
                    image_docs = await asyncio.to_thread(understand_images_with_llava, unique_images)

                all_documents.extend(image_docs)
                source_panel.update_sources(
                    images=len(image_docs),
                    chars=sum(len(d.content) for d in image_docs)
                )
                conversation.write(f"[dim]Analyzed {len(image_docs)} images successfully[/dim]")

            if not all_documents:
                conversation.write("[red]ERROR: No sources retrieved[/red]")
                source_panel.update_status("Ready")
                return

            # Step 3: Build vector DB
            source_panel.update_status("Building knowledge base...")
            conversation.write(f"[dim]Building knowledge base from {len(all_documents)} documents...[/dim]")

            with suppress_output():
                await asyncio.to_thread(self.kb.add_documents, all_documents, session_id)

            conversation.write("[dim]Knowledge base ready[/dim]")

            # Step 4: Retrieve context
            source_panel.update_status("Retrieving context...")
            conversation.write("[dim]Retrieving relevant context chunks...[/dim]")

            with suppress_output():
                retrieved_context = await asyncio.to_thread(
                    self.kb.retrieve, query, session_id, 30
                )

            source_panel.update_metrics(chunks=30)
            conversation.write(f"[dim]Retrieved {len(retrieved_context):,} chars of context[/dim]")

            if len(retrieved_context) < 100:
                conversation.write("[red]ERROR: Insufficient context[/red]")
                source_panel.update_status("Ready")
                return

            # Step 5: Generate response
            source_panel.update_status("Generating response...")
            conversation.write("")
            conversation.write("[dim]Generating response with drift detection...[/dim]")
            conversation.write("")
            conversation.write("[bold cyan]Assistant:[/bold cyan]")

            wrapper = OllamaResonanceWrapper(
                retrieved_context,
                model="llama3",
                enable_diagrams=True
            )

            token_count = 0
            response_text = ""

            for chunk in wrapper.stream_chat(query):
                conversation.write(chunk)
                response_text += chunk
                token_count += len(chunk.split())

                # Update metrics periodically
                if token_count % 10 == 0:
                    source_panel.update_metrics(
                        drift=wrapper.current_drift,
                        tokens=token_count
                    )

            # Final metrics update
            source_panel.update_metrics(
                drift=wrapper.current_drift,
                tokens=token_count
            )

            # Add summary
            conversation.write("")
            drift_color = "green" if wrapper.current_drift < 0.5 else "yellow" if wrapper.current_drift < 0.85 else "red"
            summary = f"Sources: {len(all_documents)} | Tokens: {token_count} | Drift: {wrapper.current_drift:.3f}"
            conversation.write(f"[dim]{summary}[/dim]")

            # Track assistant response in history
            self.last_assistant_response = response_text
            self.conversation_history.append(f"[ASSISTANT]\n{response_text}\n\n{summary}\n")

            source_panel.update_status("Ready")

        except Exception as e:
            conversation.write(f"[red]ERROR: {e}[/red]")
            source_panel.update_status("Error")

    def action_clear(self) -> None:
        """Clear the conversation"""
        conversation = self.query_one("#conversation", RichLog)
        conversation.clear()
        conversation.write("[green]Conversation cleared.[/green]")
        self.conversation_history = []
        self.last_assistant_response = ""

    def action_save_conversation(self) -> None:
        """Save full conversation to file"""
        if not self.conversation_history:
            conversation = self.query_one("#conversation", RichLog)
            conversation.write("[yellow]No conversation to save yet.[/yellow]")
            return

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"resonance_chat_{timestamp}.txt"

        try:
            with open(filename, "w", encoding="utf-8") as f:
                f.write("=" * 80 + "\n")
                f.write("RESONANCE GOVERNOR - CONVERSATION EXPORT\n")
                f.write(f"Saved: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write("=" * 80 + "\n\n")
                f.write("".join(self.conversation_history))

            conversation = self.query_one("#conversation", RichLog)
            conversation.write(f"[green]Conversation saved to: {filename}[/green]")
        except Exception as e:
            conversation = self.query_one("#conversation", RichLog)
            conversation.write(f"[red]Error saving: {e}[/red]")

    def action_export_last(self) -> None:
        """Export last assistant response to file"""
        if not self.last_assistant_response:
            conversation = self.query_one("#conversation", RichLog)
            conversation.write("[yellow]No assistant response to export yet.[/yellow]")
            return

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"response_{timestamp}.txt"

        try:
            with open(filename, "w", encoding="utf-8") as f:
                f.write("=" * 80 + "\n")
                f.write("RESONANCE GOVERNOR - RESPONSE EXPORT\n")
                f.write(f"Saved: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write("=" * 80 + "\n\n")
                f.write(self.last_assistant_response)

            conversation = self.query_one("#conversation", RichLog)
            conversation.write(f"[green]Response saved to: {filename}[/green]")
        except Exception as e:
            conversation = self.query_one("#conversation", RichLog)
            conversation.write(f"[red]Error exporting: {e}[/red]")


def main():
    app = ResonanceChat()
    app.run()


if __name__ == "__main__":
    main()

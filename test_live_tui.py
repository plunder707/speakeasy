#!/usr/bin/env python3
"""
Test sidecar.py + rich.Live() real-time rendering
This is the critical test for Phase 0 TUI
"""

import asyncio
import sys
from sidecar import Sidecar
from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.layout import Layout
from rich.table import Table
from rich.text import Text

try:
    import ollama
except ImportError:
    print("❌ ollama not found")
    sys.exit(1)


class LiveTUIState:
    """Shared state for live UI"""
    def __init__(self):
        self.text = ""
        self.token_count = 0
        self.status = "Starting..."


async def async_ollama_stream(prompt: str):
    """Async generator for Ollama streaming"""
    response = ollama.chat(
        model='llama3',
        messages=[{'role': 'user', 'content': prompt}],
        stream=True,
        options={'temperature': 0.7, 'num_predict': 300}
    )

    for chunk in response:
        token = chunk['message']['content']
        yield token
        await asyncio.sleep(0)


def build_layout(state: LiveTUIState) -> Layout:
    """Build the live UI layout"""
    layout = Layout()
    layout.split_column(
        Layout(name="main", ratio=8),
        Layout(name="status", size=3)
    )

    # Main panel
    content = Panel(
        Text(state.text, style="cyan"),
        title="[yellow]Real-Time Streaming Test[/yellow]",
        border_style="green"
    )
    layout["main"].update(content)

    # Status bar
    status_table = Table.grid(expand=True)
    status_table.add_column(justify="left")
    status_table.add_column(justify="right")
    status_table.add_row(
        f"[white]{state.status}[/white]",
        f"[green]Tokens: {state.token_count}[/green]"
    )
    layout["status"].update(Panel(status_table, style="blue"))

    return layout


def test_live_streaming():
    """Test real-time TUI with sidecar streaming"""
    console = Console()
    bridge = Sidecar(workers=2, name="LiveTUITest")
    state = LiveTUIState()

    prompt = "Explain the Calvin cycle in 2-3 sentences."

    try:
        with Live(build_layout(state), refresh_per_second=10, console=console) as live:
            state.status = "Streaming from Ollama..."

            # THIS IS THE CRITICAL TEST: async stream → sidecar → sync → live UI
            for token in bridge.stream(async_ollama_stream(prompt)):
                state.text += token
                state.token_count += 1
                live.update(build_layout(state))

            state.status = "Complete ✓"
            live.update(build_layout(state))

        console.print("\n[green]✅ Live TUI + Sidecar streaming: SUCCESS![/green]\n")
        return True

    except Exception as e:
        console.print(f"\n[red]❌ Failed: {e}[/red]")
        import traceback
        traceback.print_exc()
        return False

    finally:
        bridge.shutdown()


if __name__ == "__main__":
    # Check if running in interactive terminal
    if not sys.stdout.isatty():
        print("⚠️  Warning: Not running in interactive terminal (TTY)")
        print("Rich.Live() may not render properly")
        print("Run directly: python3 test_live_tui.py")
        print()

    success = test_live_streaming()
    sys.exit(0 if success else 1)

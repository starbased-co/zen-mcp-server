"""Tests for clink tool custom_system_prompt functionality."""

import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from tools.clink import CLinkTool


@pytest.fixture
def mock_registry():
    """Mock clink registry with test configuration."""
    registry = MagicMock()
    registry.list_clients.return_value = ["claude"]
    registry.list_roles.return_value = ["default"]

    mock_client = MagicMock()
    mock_client.name = "claude"
    mock_client.executable = ["claude"]
    mock_client.runner = "claude"
    mock_client.get_role.return_value = MagicMock(
        name="default", prompt_path=Path("/tmp/default_prompt.txt"), role_args=[]
    )
    registry.get_client.return_value = mock_client

    return registry


@pytest.mark.asyncio
async def test_custom_system_prompt_with_string(mock_registry):
    """Test custom_system_prompt accepts direct string."""
    with patch("tools.clink.get_registry", return_value=mock_registry):
        tool = CLinkTool()

        custom_prompt = "You are a test assistant. Be helpful."
        resolved = tool._resolve_custom_system_prompt(custom_prompt)

        assert resolved == custom_prompt


@pytest.mark.asyncio
async def test_custom_system_prompt_with_file_path(mock_registry):
    """Test custom_system_prompt reads from file path."""
    with patch("tools.clink.get_registry", return_value=mock_registry):
        tool = CLinkTool()

        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("System prompt from file.\nMultiple lines.")
            temp_path = f.name

        try:
            resolved = tool._resolve_custom_system_prompt(temp_path)
            assert resolved == "System prompt from file.\nMultiple lines."
        finally:
            Path(temp_path).unlink()


@pytest.mark.asyncio
async def test_custom_system_prompt_file_not_found(mock_registry):
    """Test custom_system_prompt raises error for non-existent file."""
    with patch("tools.clink.get_registry", return_value=mock_registry):
        tool = CLinkTool()

        with pytest.raises(Exception) as exc_info:
            tool._resolve_custom_system_prompt("/absolute/path/to/nonexistent/file.txt")

        assert "not found" in str(exc_info.value).lower()


@pytest.mark.asyncio
async def test_custom_system_prompt_none_returns_none(mock_registry):
    """Test custom_system_prompt returns None when input is None."""
    with patch("tools.clink.get_registry", return_value=mock_registry):
        tool = CLinkTool()

        resolved = tool._resolve_custom_system_prompt(None)
        assert resolved is None


@pytest.mark.asyncio
async def test_custom_system_prompt_empty_string_returns_none(mock_registry):
    """Test custom_system_prompt returns None for empty string."""
    with patch("tools.clink.get_registry", return_value=mock_registry):
        tool = CLinkTool()

        resolved = tool._resolve_custom_system_prompt("   ")
        assert resolved is None


@pytest.mark.asyncio
async def test_execute_uses_custom_system_prompt_string(mock_registry):
    """Test execute() uses custom_system_prompt string instead of role's prompt_path."""
    with patch("tools.clink.get_registry", return_value=mock_registry):
        tool = CLinkTool()

        custom_prompt_text = "Custom system prompt for this call."

        mock_agent = AsyncMock()
        mock_agent.run.return_value = MagicMock(
            parsed=MagicMock(content="Success", metadata={}),
            returncode=0,
            duration_seconds=1.0,
            stderr="",
            sanitized_command=["claude"],
            parser_name="claude",
            output_file_content=None,
        )

        with patch("tools.clink.create_agent", return_value=mock_agent):
            with patch.object(tool, "handle_prompt_file_with_fallback", return_value="Test prompt"):
                arguments = {
                    "prompt": "Test",
                    "cli_name": "claude",
                    "role": "default",
                    "custom_system_prompt": custom_prompt_text,
                }

                await tool.execute(arguments)

                mock_agent.run.assert_called_once()
                call_kwargs = mock_agent.run.call_args.kwargs
                assert call_kwargs["system_prompt"] == custom_prompt_text


@pytest.mark.asyncio
async def test_execute_uses_custom_system_prompt_file(mock_registry):
    """Test execute() reads custom_system_prompt from file path."""
    with patch("tools.clink.get_registry", return_value=mock_registry):
        tool = CLinkTool()

        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("File-based custom prompt")
            temp_path = f.name

        try:
            mock_agent = AsyncMock()
            mock_agent.run.return_value = MagicMock(
                parsed=MagicMock(content="Success", metadata={}),
                returncode=0,
                duration_seconds=1.0,
                stderr="",
                sanitized_command=["claude"],
                parser_name="claude",
                output_file_content=None,
            )

            with patch("tools.clink.create_agent", return_value=mock_agent):
                with patch.object(tool, "handle_prompt_file_with_fallback", return_value="Test prompt"):
                    arguments = {
                        "prompt": "Test",
                        "cli_name": "claude",
                        "role": "default",
                        "custom_system_prompt": temp_path,
                    }

                    await tool.execute(arguments)

                    mock_agent.run.assert_called_once()
                    call_kwargs = mock_agent.run.call_args.kwargs
                    assert call_kwargs["system_prompt"] == "File-based custom prompt"
        finally:
            Path(temp_path).unlink()


@pytest.mark.asyncio
async def test_execute_without_custom_system_prompt_uses_default(mock_registry):
    """Test execute() uses role's default prompt_path when no custom_system_prompt provided."""
    with patch("tools.clink.get_registry", return_value=mock_registry):
        tool = CLinkTool()

        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("Default role prompt")
            default_prompt_path = Path(f.name)

        try:
            mock_client = MagicMock()
            mock_client.name = "claude"
            mock_client.executable = ["claude"]
            mock_client.runner = "claude"
            mock_client.get_role.return_value = MagicMock(
                name="default", prompt_path=default_prompt_path, role_args=[]
            )
            mock_registry.get_client.return_value = mock_client

            mock_agent = AsyncMock()
            mock_agent.run.return_value = MagicMock(
                parsed=MagicMock(content="Success", metadata={}),
                returncode=0,
                duration_seconds=1.0,
                stderr="",
                sanitized_command=["claude"],
                parser_name="claude",
                output_file_content=None,
            )

            with patch("tools.clink.create_agent", return_value=mock_agent):
                with patch.object(tool, "handle_prompt_file_with_fallback", return_value="Test prompt"):
                    arguments = {
                        "prompt": "Test",
                        "cli_name": "claude",
                        "role": "default",
                    }

                    await tool.execute(arguments)

                    mock_agent.run.assert_called_once()
                    call_kwargs = mock_agent.run.call_args.kwargs
                    assert call_kwargs["system_prompt"] == "Default role prompt"
        finally:
            default_prompt_path.unlink()

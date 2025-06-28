"""
Tests for the init command.
"""
from typing import AnyStr

import pytest

from prompt_mixer.commands.init import do_init
from prompt_mixer.gateways.git import GitGateway
from prompt_mixer.config import Config


@pytest.fixture
def mock_console(mocker):
    return mocker.MagicMock()


@pytest.fixture
def mock_path(tmp_path):
    return tmp_path / "prompt-mixer"

@pytest.fixture
def mock_config(mock_path):
    return Config(mock_path)


@pytest.fixture
def mock_git_gateway(mocker):
    return mocker.MagicMock(spec=GitGateway)


class DescribeDoInit:

    def should_clone_repository_when_remote_is_provided(self, mock_git_gateway, mock_console, mock_path, mock_config):
        remote = "git@github.com:user/repo.git"
        mock_git_gateway.clone.return_value = (True, AnyStr)

        do_init(mock_console, mock_config, remote, None, None, mock_git_gateway)

        mock_git_gateway.clone.assert_called_once_with(remote, mock_path)
        assert mock_console.print.call_count >= 3  # At least 3 print calls

    def should_initialize_new_repository_when_remote_not_provided(self, mock_git_gateway, mock_console, mock_path, mock_config):
        mock_git_gateway.init.return_value = (True, AnyStr)

        do_init(mock_console, mock_config, None, None, None, mock_git_gateway)

        mock_git_gateway.init.assert_called_once_with(mock_path)
        assert mock_console.print.call_count >= 3  # At least 3 print calls

    def should_create_directory_if_it_doesnt_exist(self, mock_git_gateway, mock_console, mock_path, mock_config):
        mock_git_gateway.init.return_value = (True, AnyStr)

        do_init(mock_console, mock_config, None, None, None, mock_git_gateway)

        assert mock_path.exists()
        assert mock_path.is_dir()

    def should_handle_provider_and_model_parameters(self, mock_git_gateway, mock_console, mock_path, mock_config):
        provider = "ollama"
        model = "phi3"
        mock_git_gateway.init.return_value = (True, AnyStr)

        do_init(mock_console, mock_config, None, provider, model, mock_git_gateway)

        mock_git_gateway.init.assert_called_once_with(mock_path)
        message_found = False
        for args, _ in mock_console.print.call_args_list:
            if "Provider and model configuration" in str(args):
                message_found = True
                break
        assert message_found

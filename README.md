# Technical Interview

This repository contains the base code for technical interviews at Mirico.

## Installing

The recommended way to use this repository is through GitHub Codespaces or Dev Containers. To get started, simply open this repository in GitHub Codespaces or clone it locally and open it in VS Code with the Dev Containers extension installed.

There is no requirement to use Codespaces or Dev Containers, but they provide a convenient way to set up a consistent development environment.

Free GitHub accounts should have a sufficient number of Codespaces hours to complete the assignment. Dev Containers can be used locally without any additional cost.

If you wish to install any additional python packages, please add them using `uv add <package-name>` so that they are included in the `pyproject.toml` file.

If you prefer to set up the environment manually, you will need to have a recent python version installed. If you use any dependencies not included in the `pyproject.toml`, please make sure to note them in your submission, or add them to the `pyproject.toml` file.

### Video of setting up a Codespace

https://github.com/user-attachments/assets/0ec92ce6-0dd8-4281-82a2-dfb010036e1f



## Challenge

The take-home assignment is located in the `challenge` folder. Read the [assignment_instructions.md](challenge/assignment_instructions.md) file in that folder for more information about the take-home assignment.

## AI Usage

You are permitted to use AI tools such as GitHub Copilot, ChatGPT, or others to assist you in completing the assignment. However, please be transparent about your usage of these tools in your submission. Specifically, please include a brief summary of how you used AI tools, if at all, and any significant contributions they made to your code. Further, you will be expected to explain and discuss your code during the interview, so please ensure you understand your submission in full.

## Submitting your work

You have multiple options for submitting your work:
1. If you used GitHub Codespaces, you can simply push your changes to a fork of this repository and share the link with us. This is also the reccomended option if you used Dev Containers locally. Do not make a pull request; just share the link to your forked repository.
2. If you would prefer not to make a public repository on GitHub, you can send us the diff of your changes as a patch file. You can create a patch file by running the following command from the root of the repository:
   ```bash
   git diff > submission.patch
   ```
   You can then send us the `submission.patch` file.
3. If neither of the above options are suitable, you can also create a zip/tar file of the repository with your changes and send that to us.

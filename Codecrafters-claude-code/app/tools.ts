import type OpenAI from "openai";
import { readFile, writeFile, mkdir } from "node:fs/promises";
import { dirname } from "node:path";
import { exec } from "node:child_process";
import { promisify } from "node:util";

const execAsync = promisify(exec);

export const TOOLS = [
    {
        type: "function",
        function: {
            name: "Read",
            description: "Read and return the contents of a file",
            parameters: {
                type: "object",
                required: ["file_path"],
                properties: {
                    file_path: { type: "string", description: "The path to the file to read" },
                },
            },
        },
    },
    {
        type: "function",
        function: {
            name: "Write",
            description: "Write content to a file",
            parameters: {
                type: "object",
                required: ["file_path", "content"],
                properties: {
                    file_path: { type: "string", description: "The path of the file to write to" },
                    content: { type: "string", description: "The content to write to the file" },
                },
            },
        },
    },
    {
        type: "function",
        function: {
            name: "Bash",
            description: "Execute a shell command",
            parameters: {
                type: "object",
                required: ["command"],
                properties: {
                    command: { type: "string", description: "The command to execute" },
                },
            },
        },
    },
] as const satisfies OpenAI.ChatCompletionTool[];

export type ToolName = (typeof TOOLS)[number]["function"]["name"];

type Handler = (rawArgs: string) => Promise<string>;

interface ReadArgs {
    file_path: string;
}

interface WriteArgs {
    file_path: string;
    content: string;
}

interface BashArgs {
    command: string;
}

export const HANDLERS: Record<ToolName, Handler> = {
    async Read(rawArgs) {
        const { file_path: filePath } = JSON.parse(rawArgs) as ReadArgs;
        return await readFile(filePath, "utf-8");
    },

    async Write(rawArgs) {
        const { file_path: filePath, content } = JSON.parse(rawArgs) as WriteArgs;
        await mkdir(dirname(filePath), { recursive: true });
        await writeFile(filePath, content, "utf-8");
        return `Successfully wrote ${content.length} characters to ${filePath}`;
    },

    async Bash(rawArgs) {
        const { command } = JSON.parse(rawArgs) as BashArgs;
        try {
            const { stdout, stderr } = await execAsync(command);
            return (stdout + stderr).trim() || "(command succeeded with no output)";
        } catch (error) {
            const failure = error as { code?: number; stdout?: string; stderr?: string };
            const output = `${failure.stdout ?? ""}${failure.stderr ?? ""}`.trim();
            return `Command failed (exit code ${failure.code}):\n${output}`.trim();
        }
    },
};

export function findHandler(name: string): Handler | undefined {
    return (HANDLERS as Record<string, Handler | undefined>)[name];
}

import OpenAI from "openai";
import { TOOLS } from "./tools";
import { readFile, writeFile, mkdir } from "node:fs/promises";
import { dirname } from "node:path";
import { exec } from "node:child_process";
import { promisify } from "node:util";

async function main() {
    const [, , flag, prompt] = process.argv;
    const apiKey = process.env.OPENROUTER_API_KEY;
    const baseURL = process.env.OPENROUTER_BASE_URL ?? "https://openrouter.ai/api/v1";
    const execAsync = promisify(exec);

    if (!apiKey) {
        throw new Error("OPENROUTER_API_KEY is not set");
    }
    if (flag !== "-p" || !prompt) {
        throw new Error("error: -p flag is required");
    }

    const client = new OpenAI({
        apiKey: apiKey,
        baseURL: baseURL,
    });

    const messages: OpenAI.ChatCompletionMessageParam[] = [{ role: "user", content: prompt }];

    const MAX_TURNS = 10;

    for (let turn = 0; turn < MAX_TURNS; turn++) {
        const response = await client.chat.completions.create({
            model: "anthropic/claude-haiku-4.5",
            messages: messages,
            tools: TOOLS,
        });

        if (!response.choices || response.choices.length === 0) {
            throw new Error("no choices in response");
        }

        const message = response.choices[0].message;
        const calls = message.tool_calls;
        messages.push(message);

        if (!calls || calls.length === 0) {
            console.log(message.content);
            break;
        }

        for (const tool_call of calls) {
            if (tool_call.type !== "function") {
                throw new Error("tool call is not a function");
            }

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

            const tool_call_id = tool_call.id;
            const function_name = tool_call.function.name;
            const function_args = tool_call.function.arguments;
            let result: string;

            switch (function_name) {
                case "Read": {
                    const args = JSON.parse(function_args) as ReadArgs;
                    try {
                        result = await readFile(args.file_path, "utf-8");
                    } catch (error) {
                        result = `Error reading file ${args.file_path}: ${error instanceof Error ? error.message : String(error)}`;
                        console.error(`[tool] ${function_name} failed: ${result}`);
                    }
                    break;
                }
                case "Write": {
                    const args = JSON.parse(function_args) as WriteArgs;
                    try {
                        await mkdir(dirname(args.file_path), {
                            recursive: true,
                        });
                        await writeFile(args.file_path, args.content, "utf-8");
                        result = `Successfully wrote ${args.content.length} characters to ${args.file_path}`;
                    } catch (error) {
                        result = `Error writing ${args.file_path}: ${error instanceof Error ? error.message : String(error)}`;
                        console.error(`[tool] ${result}`);
                    }
                    break;
                }
                case "Bash": {
                    const args = JSON.parse(function_args) as BashArgs;
                    try {
                        const { stdout, stderr } = await execAsync(args.command);
                        result = (stdout + stderr).trim() || "(command succeeded with no output)";
                    } catch (error) {
                        const e = error as {
                            code?: number;
                            stdout?: string;
                            stderr?: string;
                        };
                        result = `Command failed (exit code ${e.code}):\n${e.stdout ?? ""}${e.stderr ?? ""}`.trim();
                        console.error(`[tool] ${result}`);
                    }
                    break;
                }
                default:
                    throw new Error(`Unknown function name: ${function_name}`);
            }

            messages.push({
                role: "tool",
                tool_call_id: tool_call_id,
                content: result,
            });
        }
    }
}

main();

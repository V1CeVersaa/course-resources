import OpenAI from "openai";
import { TOOLS, findHandler } from "./tools";

const MAX_TURNS = 10;

function errorMessage(error: unknown): string {
    return error instanceof Error ? error.message : String(error);
}

async function main() {
    const [, , flag, prompt] = process.argv;
    const apiKey = process.env.OPENROUTER_API_KEY;
    const baseURL = process.env.OPENROUTER_BASE_URL ?? "https://openrouter.ai/api/v1";

    if (!apiKey) {
        throw new Error("OPENROUTER_API_KEY is not set");
    }
    if (flag !== "-p" || !prompt) {
        throw new Error("error: -p flag is required");
    }

    const client = new OpenAI({ apiKey, baseURL });

    const messages: OpenAI.ChatCompletionMessageParam[] = [{ role: "user", content: prompt }];

    for (let turn = 0; turn < MAX_TURNS; turn++) {
        const response = await client.chat.completions.create({
            model: "anthropic/claude-haiku-4.5",
            messages,
            tools: TOOLS,
        });

        if (!response.choices || response.choices.length === 0) {
            throw new Error("no choices in response");
        }

        const message = response.choices[0].message;
        messages.push(message);

        const toolCalls = message.tool_calls;
        if (!toolCalls || toolCalls.length === 0) {
            console.log(message.content);
            return;
        }

        for (const toolCall of toolCalls) {
            if (toolCall.type !== "function") {
                throw new Error("tool call is not a function");
            }

            const { name, arguments: rawArgs } = toolCall.function;
            const handler = findHandler(name);

            let result: string;
            try {
                result = handler ? await handler(rawArgs) : `Unknown tool: ${name}`;
            } catch (error) {
                result = `Tool ${name} failed: ${errorMessage(error)}`;
                console.error(`[tool] ${result}`);
            }

            messages.push({ role: "tool", tool_call_id: toolCall.id, content: result });
        }
    }

    throw new Error(`agent loop exceeded ${MAX_TURNS} turns without a final answer`);
}

main();

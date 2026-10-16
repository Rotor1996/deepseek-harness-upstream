/** Opt-in Grok gateway smoke for tool calls, optional reasoning, and history replay. */
import { afterEach, describe, expect, it } from 'vitest'
import { Context } from '@deepseek-ai/cordis'
import LlmRuntime, { createToolResultMessage, createUserMessage, ReasoningEffortId } from '@deepseek-ai/dsh-llm'
import * as LlmPiAi from '@deepseek-ai/dsh-llm-pi-ai'
import { assemble } from './assemble.ts'

const baseURL = process.env.DSH_GROK_BASE_URL
const model = process.env.DSH_GROK_MODEL
let ctx: Context | undefined

afterEach(async () => { await ctx?.fiber.dispose(); ctx = undefined })

describe.skipIf(!baseURL || !model || !process.env.DSH_GROK_API_KEY)('Grok gateway history', () => {
  it('replays a real tool round and any returned reasoning', async () => {
    ctx = new Context()
    await ctx.plugin(LlmRuntime)
    await ctx.plugin(LlmPiAi, { providers: { gateway: {
      api: 'openai-completions', baseURL: baseURL!, apiKeyEnv: 'DSH_GROK_API_KEY',
      models: [{ id: model!, parseThinkTags: true, reasoningEfforts: { low: 'low' } }],
    } } })
    const user = createUserMessage({ source: { kind: 'user' }, content: [{ type: 'text', text:
      'Call load_workspace_dependencies exactly once with {}. After receiving its result, reply exactly GROK_TAGS_OK.' }] })
    const tools = [{ name: 'load_workspace_dependencies', description: 'Locate available Python runtime and installed libraries for file creation.', parameters: {
      type: 'object', properties: {}, additionalProperties: false,
    } }]
    const first = await assemble(ctx, { provider: 'gateway', model: model!, reasoningEffort: ReasoningEffortId('low'),
      maxTokens: 2048, messages: [user], tools })
    expect(first.finish.kind).toBe('tool-calls')
    const calls = first.message.content.filter(block => block.type === 'tool-call')
    expect(calls).toHaveLength(1)
    expect(calls[0]?.name).toBe('load_workspace_dependencies')
    expect(calls[0]?.arguments).toBe('{}')
    expect(first.message.source).toMatchObject({ kind: 'model', replayState: {
      blocks: first.message.content.map(block => ({ type: block.type })),
    } })
    expect(first.message.content.filter(block => block.type === 'text').map(block => block.text).join('')).not.toContain('<think>')
    const second = await assemble(ctx, { provider: 'gateway', model: model!, reasoningEffort: ReasoningEffortId('low'),
      maxTokens: 2048, messages: [user, first.message, createToolResultMessage({
        callId: calls[0]!.id, content: [{ type: 'text', text: 'No image generation or Python runtime is available.' }], isError: false,
      }), createUserMessage({ source: { kind: 'user' }, content: [{ type: 'text', text: 'Do not call tools again. Reply exactly GROK_TAGS_OK.' }] })], tools })
    expect(second.finish.kind).toBe('stop')
    expect(second.message.content.filter(block => block.type === 'text').map(block => block.text).join('').trim()).toBe('GROK_TAGS_OK')
    const reasoningCount = [...first.message.content, ...second.message.content].filter(block => block.type === 'reasoning').length
    console.info('Grok gateway smoke reasoning blocks:', reasoningCount, reasoningCount === 0 ? '(reasoning parsing not exercised)' : '')
  })
})

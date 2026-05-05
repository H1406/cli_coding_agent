# Conversation and Repo Context Plan

This document describes how to add persistent conversation memory and repository retrieval to the CLI coding agent without implementing it yet.

## Goal

Build a context system that:

- stores user conversations durably
- scans and chunks the repository
- embeds repository chunks into a vector database
- optionally embeds conversation summaries or selected turns
- retrieves both conversation context and repo context for each agent run
- assembles a bounded prompt so context stays useful and affordable

## Recommended architecture

Use two storage layers instead of forcing one database to do everything:

1. Relational database for source-of-truth records
2. Vector database for semantic retrieval

Recommended default stack:

- Relational database: PostgreSQL
- Vector store: `pgvector` in the same PostgreSQL instance for the first production version

Why this is a good default:

- one deployment target is simpler than running a separate vector service
- PostgreSQL is reliable for conversation history, metadata, and indexing
- `pgvector` is good enough for a coding agent until scale becomes very large

If you want a lighter local-only MVP:

- SQLite for conversations
- LanceDB or Chroma for vectors

That option is easier for local prototyping, but PostgreSQL plus `pgvector` is the cleaner long-term deployment path.

## What to store

Store raw data and derived data separately.

### Relational tables

Recommended logical entities:

- `sessions`
- `messages`
- `message_attachments`
- `repo_index_runs`
- `repo_files`
- `repo_chunks`
- `conversation_summaries`
- `retrieval_events`

Suggested meaning:

- `sessions`: one conversation thread or work session
- `messages`: every user and assistant turn
- `message_attachments`: optional artifacts, file references, command results, tool traces
- `repo_index_runs`: each indexing pass over a repository snapshot
- `repo_files`: tracked files and metadata such as path, hash, language, size
- `repo_chunks`: chunk metadata before or alongside vector storage
- `conversation_summaries`: rolling summaries to avoid retrieving every message verbatim
- `retrieval_events`: what was retrieved and why, for debugging and evaluation

### Vector collections

Keep at least two logical vector namespaces:

- `repo_context`
- `conversation_context`

Each vector record should have metadata such as:

- `source_type`
- `session_id`
- `repo_id`
- `file_path`
- `chunk_id`
- `message_id`
- `language`
- `branch`
- `commit_sha`
- `created_at`
- `token_count`

## Retrieval design

Do not retrieve the entire repo or the entire conversation every turn.

Use a layered strategy:

1. Recent conversational context
2. Conversation summary context
3. Retrieved semantic conversation memory
4. Retrieved semantic repo context
5. Optional deterministic file context based on the current working files

### Recommended retrieval flow per request

1. Load the latest `N` conversation messages from the active session.
2. Load the latest rolling summary for the session, if it exists.
3. Embed the new user query.
4. Search conversation vectors for relevant prior work.
5. Search repo vectors for relevant code or docs.
6. Merge and rerank the results.
7. Deduplicate overlapping chunks.
8. Trim the final context to a token budget.
9. Build the final model prompt with clear source labels.

## Step-by-step plan

## Step 1: Decide your deployment mode

Choose one of these paths before building:

- Local-first MVP: SQLite plus LanceDB or Chroma
- Production-ready baseline: PostgreSQL plus `pgvector`

Recommendation:

- If this project will be deployed for real users, start with PostgreSQL plus `pgvector`.
- If this is only for rapid local experimentation, start with SQLite plus a local vector store and migrate later.

## Step 2: Define the data model first

Before touching embeddings, define the relational schema.

Minimum fields to include:

- `sessions`: `id`, `title`, `created_at`, `updated_at`
- `messages`: `id`, `session_id`, `role`, `content`, `created_at`, `sequence_no`
- `repo_files`: `id`, `repo_id`, `path`, `content_hash`, `language`, `size_bytes`, `indexed_at`
- `repo_chunks`: `id`, `repo_file_id`, `chunk_index`, `content`, `token_count`, `start_line`, `end_line`
- `conversation_summaries`: `id`, `session_id`, `summary_text`, `up_to_message_id`, `created_at`

Recommendation:

- store raw message content in the relational database even if you also embed it
- do not use vectors as the only source of truth

## Step 3: Design the repository indexing pipeline

The repo indexing system should:

1. walk the repo
2. ignore excluded files and directories
3. hash file contents
4. detect changed files only
5. chunk files intelligently
6. generate embeddings
7. upsert vectors and metadata

Recommended exclusions:

- `.git/`
- `node_modules/`
- `.venv/`
- build artifacts
- binary files
- lockfiles only if they are not useful for retrieval

Recommendation:

- use incremental indexing based on file hash, not full reindexing every run
- store `commit_sha` or an equivalent repo snapshot marker with each index run

## Step 4: Chunk the repository carefully

Chunking quality matters more than many people expect.

Recommended chunking rules:

- chunk by syntax-aware boundaries when possible
- prefer function, class, or section boundaries
- use overlap between chunks
- keep metadata pointing back to file path and line ranges

Practical guidelines:

- aim for roughly 300 to 800 tokens per chunk
- use 50 to 120 tokens of overlap
- keep one chunk from mixing unrelated code regions when possible

For docs and markdown:

- chunk by headings and paragraphs

For code:

- chunk by AST-aware units when your language tooling allows it
- otherwise chunk by lines and nearest symbol boundaries

## Step 5: Decide what conversation content should be embedded

Do not embed every raw turn forever without a policy.

Recommended strategy:

- store all raw turns in the relational database
- embed selected user messages and assistant responses
- create rolling summaries for long sessions
- embed the summaries too

Good candidates for embedding:

- user requirements
- accepted plans
- decisions and constraints
- assistant explanations that define future behavior
- tool results that materially changed the state of the work

Avoid embedding:

- trivial acknowledgements
- duplicated content
- large noisy logs unless summarized first

## Step 6: Add a summarization layer

Conversation memory gets better when you summarize old turns.

Recommended pattern:

- keep the last few raw messages directly
- periodically summarize older messages into a structured summary
- store both the summary text and the message range it covers
- retrieve summaries before digging into old raw turns

A useful summary format:

- current task
- decisions made
- files touched
- unresolved questions
- constraints or user preferences

## Step 7: Define your embedding policy

Pick one embedding model and keep it consistent within an index version.

Recommendations:

- version your embeddings
- store the embedding model name with each vector
- reindex selectively when you change chunking or model versions

Do not mix embeddings from different models in the same retrieval path unless you know exactly how you will handle it.

## Step 8: Build the retrieval and ranking pipeline

Vector similarity alone is usually not enough.

Recommended ranking approach:

1. semantic similarity search
2. metadata filtering
3. recency weighting for conversation results
4. path or symbol weighting for repo results
5. deduplication
6. optional reranking step

Useful filters:

- current session only for conversational retrieval, unless cross-session memory is explicitly enabled
- current repository only
- current branch or latest indexed branch
- language or file type

Recommendation:

- give recent conversation messages a deterministic slot in the prompt
- use vector retrieval to supplement, not replace, the conversation window

## Step 9: Assemble the final prompt with sections

When you build the prompt, label each context source clearly.

Recommended prompt sections:

1. system instructions
2. conversation summary
3. recent conversation turns
4. retrieved conversation memories
5. retrieved repo context
6. current user request

Recommendation:

- include citations or source labels such as file path, line range, message timestamp, and session id
- this makes debugging and evaluation much easier

## Step 10: Add freshness and sync rules

You need a policy for when repo embeddings become stale.

Recommended triggers:

- on startup, check whether the repo head changed
- after file edits, mark changed files as dirty
- reindex changed files after each tool run or before the next retrieval

Recommendation:

- do not re-embed the full repo after every edit
- only re-embed changed files and delete stale vectors for removed files

## Step 11: Add security and privacy rules

Conversation memory and repo memory can contain sensitive data.

Recommendations:

- encrypt database storage where appropriate
- do not embed secrets, tokens, or `.env` files
- exclude private keys and machine credentials from indexing
- define retention rules for user conversations
- allow per-user deletion or session deletion

## Step 12: Evaluate retrieval quality before trusting it

Add a manual evaluation pass early.

Create test scenarios such as:

- user asks about a decision made earlier in the same session
- user asks about a file changed yesterday
- user references a symbol that exists in a deeply nested file
- repo contains many similar utility files

Measure:

- whether the correct chunks were retrieved
- whether noisy chunks displaced useful ones
- whether summaries preserved the important decisions

## Recommended implementation order

Build in this order:

1. relational conversation storage
2. repo file scanning and chunk metadata
3. vector storage for repo chunks
4. repo retrieval
5. conversation summaries
6. conversation embeddings
7. combined retrieval and reranking
8. observability and retrieval evaluation

This order keeps the system debuggable while complexity is still manageable.

## Strong recommendations

- Start with PostgreSQL plus `pgvector` unless you are certain this is a throwaway prototype.
- Keep raw conversations in SQL and treat vectors as a retrieval index, not as your memory source of truth.
- Use incremental repo indexing based on file hashes.
- Summarize long conversations instead of embedding every turn blindly.
- Retrieve recent raw turns and summary context first, then semantic memories, then repo chunks.
- Log retrieval decisions so you can inspect why the agent answered the way it did.
- Put strict token budgets around the final context assembly.

## Common mistakes to avoid

- storing only embeddings and losing the raw source text
- embedding the entire conversation history every turn
- indexing generated artifacts and noisy directories
- retrieving too many similar chunks from the same file
- treating vector search as perfect relevance
- skipping summaries for long sessions
- failing to version embedding models and chunking strategies

## Suggested first milestone

Your first milestone should be:

- conversations saved in SQL
- repository files chunked and embedded
- a query can retrieve top conversation summaries plus top repo chunks
- the final prompt shows exactly which sources were included

Once that works well, then add more advanced ranking, caching, and background indexing.

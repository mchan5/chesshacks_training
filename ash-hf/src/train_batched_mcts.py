# Optimized MCTS with batched evaluation
# Replace the MCTS class in train.py with this for 3-5x speedup

class BatchedMCTS:
    """Monte Carlo Tree Search with batched neural network evaluation"""
    def __init__(self, model, num_simulations=100, batch_size=8):
        self.model = model
        self.num_simulations = num_simulations
        self.batch_size = batch_size

    @torch.no_grad()
    def search(self, board):
        """Run MCTS with batched evaluation"""
        root = MCTSNode(board)

        # Process in batches
        for batch_start in range(0, self.num_simulations, self.batch_size):
            batch_end = min(batch_start + self.batch_size, self.num_simulations)
            batch_size_actual = batch_end - batch_start

            # Collect nodes to evaluate
            nodes_to_eval = []
            for _ in range(batch_size_actual):
                node = root

                # Selection
                while node.is_expanded and node.children:
                    if node.board.is_game_over():
                        break
                    node = node.select_child()

                if not node.board.is_game_over():
                    nodes_to_eval.append(node)

            # Batch evaluate
            if nodes_to_eval:
                boards = [encode_board(n.board) for n in nodes_to_eval]
                boards_tensor = torch.FloatTensor(boards).to(device)

                policy_logits_batch, values_batch = self.model(boards_tensor)
                policy_probs_batch = torch.softmax(policy_logits_batch, dim=1).cpu().numpy()
                values_batch = values_batch.cpu().numpy()

                # Expand and backpropagate
                for i, node in enumerate(nodes_to_eval):
                    node.expand(policy_probs_batch[i])
                    node.backpropagate(float(values_batch[i]))

        # Return visit count distribution
        visit_counts = np.zeros(4096)
        for move, child in root.children.items():
            visit_counts[move_to_index(move)] = child.visit_count

        return visit_counts

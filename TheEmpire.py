import random
import hlt
import math
from hlt import NORTH, EAST, SOUTH, WEST, STILL, Move

# Constants
DEGRADATION_RATE          = math.exp(-1 / random.uniform(2.5, 3.5))
DIAGONAL_DEGRADATION_RATE = DEGRADATION_RATE * DEGRADATION_RATE

NEIGH_DIRECTIONS = {NORTH, SOUTH, WEST, EAST}
SE, SW, NW, NE   = range(5, 9)

NEUTRAL_ID   = 0
MAX_STRENGTH = 255


def compute_best_move_square(square, avail_dirs=NEIGH_DIRECTIONS):
    """
    Get best position for a square without taking into consideration other square from his territory
    @param square:          Current square
    @param avail_dirs:      Directions that need to be computed in actual conditions
    @return:                Best individual move for a square considering all distances to our bot's borders
    """

    # If we don't have any good moves left, stay still
    if not avail_dirs:
        return Move(square, STILL)

    if len(avail_dirs) == 1:
        best_opportunity_direction = list(avail_dirs)[0]
    else:
        # Get and unite additional diagonal directions
        diagonal_directions = get_diagonal_directions(avail_dirs)
        all_directions      = avail_dirs.union(diagonal_directions)
        
        # Calculate best direction to go with the future in mind
        best_opportunity_value      = -1
        best_opportunity_direction  = None
        for direction in all_directions:
            curr_opportunity_value  = move_opportunity(direction, square)
            if curr_opportunity_value > best_opportunity_value:
                best_opportunity_value      = curr_opportunity_value
                best_opportunity_direction  = direction
    
        # If the best direction is a diagonal one then calculate the best first move
        # (eg. NE -> First N or first E)
        if best_opportunity_direction >= 5:
            best_opportunity_direction = get_best_path_direction(square, best_opportunity_direction)

    target_square   = game_map.get_target(square, best_opportunity_direction)
    # Using the uniform distribution, we introduce a stochastic behaviour in the system
    # `distance_factor` is a way to measure if it would be worth to bring the production
    # from the center of our territory to the edges (to our frontier)
    distance_factor = random.uniform(4, 6) * math.exp(-get_distance_to_frontier(square) / (random.uniform(8, 12)))

    # Decide if we want to move or stay still
    if  (square.strength <= distance_factor * square.production and target_square.owner == my_id) or \
        (not is_valid_move(square, target_square)):
        return Move(square, STILL)
    
    return Move(square, best_opportunity_direction)


def get_distance_to_frontier(square):
    """
    Minimum distance to our frontier following one of the four cardinal direction
    @param square: Current square
    @return:       Minimum distance to frontier
    """

    def get_curr_distance_to_frontier(neighbor, direction):
        """
        Compute the distance from the square `neighbor` to the frontier in the direction `direction`
        @param neighbor:  The neighbour of the current square
        @param direction: Current direction to check
        @return:          Distance to the frontier following a direction (max of 0.8 of the boards width)
        """
        curr_distance = 0
        while   neighbor.owner == my_id and \
                curr_distance <= random.randint(int (0.75 * WIDTH), int (0.85 * WIDTH)):
            neighbor = game_map.get_target(neighbor, direction)
            curr_distance += 1
        return curr_distance
  
    # Initialize the `min_distance` with the map's width
    min_distance = WIDTH
    
    # Compute the minimum distance to the closest border
    for direction in NEIGH_DIRECTIONS:
        neighbor     = game_map.get_target(square, direction)
        min_distance = min(min_distance, get_curr_distance_to_frontier(neighbor, direction))
        
    return min_distance


def get_best_path_direction(square, diag_direction):
    """
    Computes the best path to move from `square` to a given diagonal direction
    We can first go through a vertical (N/S) direction or through a horizontal one (W/E)
    @param square:          Current square
    @param diag_direction:  The diagonal direction to follow
    @return:                The preferable order to go through the neighbour squares
    """
    
    # Convert a diagonal direction to 2 `neigh_directions`
    if   diag_direction == NE:
        neigh_directions = [NORTH, EAST]
    elif diag_direction == NW:
        neigh_directions = [NORTH, WEST]
    elif diag_direction == SW:
        neigh_directions = [SOUTH, WEST]
    elif diag_direction == SE:
        neigh_directions = [SOUTH, EAST]
    
    # Get the two possible target squares
    targets = []
    for direction in neigh_directions:
        targets.append(game_map.get_target(square, direction))

    # Try to go through a square that I already own (with small strength) if it can
    # Otherwise, go through the square with the least amount of strength
    strengths = []
    if targets[0].owner == my_id:
        strengths.append(-(MAX_STRENGTH - targets[0].strength))
    else:
        strengths.append(targets[0].strength)

    if targets[1].owner == my_id:
        strengths.append(-(MAX_STRENGTH - targets[1].strength))
    else:
        strengths.append(targets[1].strength)

    # Select the best neighbour the move through
    if strengths[0] < strengths[1]:
        return neigh_directions[0]

    return neigh_directions[1]


def get_diagonal_directions(directions):
    """
    Starting from the given neighboring `directions`, compute all the valid diagonal directions
    @param directions:   List of `neigh directions`
    @return:             Set contains the diagonal directions derived from the list of `neigh directions`
    """
    
    diag_directions = []

    if NORTH in directions and WEST in directions:
        diag_directions.append(NW)
    if SOUTH in directions and WEST in directions:
        diag_directions.append(SW)
    if NORTH in directions and EAST in directions:
        diag_directions.append(NE)
    if SOUTH in directions and EAST in directions:
        diag_directions.append(SE)

    return set(diag_directions)
    

def get_target(square, direction):
    """
    Starting from the current `square`, return the `target` at the position `square.direction + direction`
    @param square:       Current square
    @param direction:    The direction from which we want to take the target 
    @return:             Target square
    """
    return get_target_diagonal(direction, square) if (direction >= 5) else game_map.get_target(square, direction)


def get_target_diagonal(direction, square):
    """
    Get the target located at the given `direction`, starting from `square`
    @param direction:   The diagonal direction in which to go    
    @param square:      Current square
    @return:            The target square gotten from going into that direction
    """
    
    if   direction == SW:
        return game_map.get_target(game_map.get_target(square, SOUTH), WEST)
    elif direction == SE:
        return game_map.get_target(game_map.get_target(square, SOUTH), EAST)
    elif direction == NE:
        return game_map.get_target(game_map.get_target(square, NORTH), EAST)
    elif direction == NW:
        return game_map.get_target(game_map.get_target(square, NORTH), WEST)


def is_valid_move(source, target):
    """
    Checks if there is a valid move from `source` to `target`
    @param source_square:       Starting square
    @param target_square:       Ending (target) square
    @return:                    True if the move is possible, False otherwise
    """
    
    # If the square I want to move to is my square then it is a valid move
    if target.owner == my_id:
        return True
    # Checks if the square I am fighting for is neutral
    elif target.owner == NEUTRAL_ID: 
        # If an enemy would overkill their own squares to capture the current square
        # then it is a move I want to make
        overkill = get_overkill_factor(source, target)
        # The overkill factor is relative to the maximum strength of a square
        if overkill >= 1 or source.strength > target.strength:
            return True
    else:
        # If the square I am fighting for is an enemy's square
        if source.strength >= target.strength:
            return True

    # The move is not possible
    return False


def square_opportunity(source, target):
    """
    Computes the opportunity to capture `target` starting from `source`
    @param source:       Starting square
    @param target:       Ending (target) square
    @return:             Normalized opportunity factor
    """

    # This factor (which represent square ownership) refers to the fact that
    # in the beginning when there are multiple bots we don't want to fight so much (we let them fight)
    # and only when there is one remaining enemy bot we give more "value" to fighting it
    opportunity_factor  = get_opportunity_factor(target)

    # This factor (which influences how "good" a target is) is the product
    # between the `production` and the `dificulty` (inverse of strength) of the target square
    # The `production` in this case has more "weight" than the strength of the square because
    # we value `production` over time over the initial cost of breaking it
    # We also add stochastic behaviour because in practice we noticed it works better
    # The strength and production of a square are normalised
    opportunity         = (target.production / MAX_PRODUCTION) * \
                          ((1 - (target.strength / MAX_STRENGTH)) ** random.randint(2, 4))

    # The difficulty of conquering a square 'target' starting from a square 'source' with
    # a production of `source.production`
    ratio_source_target = max(target.strength - source.strength, 0.0) / max(1, source.production)

    # The last factor which cushions the product of the other 2 factors
    # For small values of the ratio result big values for the target factor, this means that it is
    # opportune to conquer a square `target` if it has a small strength and we get a big production on
    # square `source`
    source_target_factor = math.exp(-ratio_source_target / random.uniform(1.8, 2.2))

    return opportunity * opportunity_factor * source_target_factor


def move_opportunity(direction, square):
    """
    Computes the total opportunity to move from the current `square` in the given `direction`
    @param direction:   Direction through which we want to move
    @param square:      Current square
    @return:            Opportunity of moving to direction from current square
    """

    # Decides if it is a diagonal move 
    is_diagonal = direction >= 5

    # If the move is diagonal, we have to multiply each move with the degradation factor
    degradation_rate = DIAGONAL_DEGRADATION_RATE if is_diagonal else DEGRADATION_RATE
    horizon          = DIAGONAL_RADAR            if is_diagonal else RADAR

    def get_total_opportunity(direction, square):
        opportunity    = 0
        current_square = square
        
        for i in range(horizon):
            neighbor = get_target(current_square, direction)

            # At each step, it multiplies by the degradation factor (the farther the square,
            # the lower the interest rate, the less it influences the opportunity)
            opportunity   += square_opportunity(square, neighbor) * pow(degradation_rate, i)
            current_square = neighbor

        return opportunity
    
    return get_total_opportunity(direction, square)


def get_opportunity_factor(square):
    """"
    Computes a factor that takes into account square owner
    @param square_owner:    Current square
    @return:                Computed opportunity factor
    """
    
    # If it is my square, then I don't need to capture it (return 0.0 factor)
    if square.owner == my_id:
        return 0.0
    
    # If it is a neutral square, I will try to capture it with a factor of 1.0
    if square.owner == NEUTRAL_ID:
        return 1.0
    
    # If there are more enemies, try to be more reserved
    if number_of_enemy_bots == 1:
        return random.uniform(1.1, 1.3)
    else:
        return random.uniform(0.6, 0.8)


def get_overkill_factor(source, target):
    """
    Computes a factor taking into account the overkill damage which
    would result from moving `source_square` to `target_square
    @param source:    Starting square
    @param target:    Ending (target) square
    @return:          The overkill rate that we get if we conquere a square 
    """
    overkill_damage = 0

    # Get the neigbours of the `source` square and create a list of `enemy` squares
    neighbors = game_map.neighbors(target)
    enemies = []
    for neighbour in neighbors:
        if neighbour.owner != my_id and neighbour.owner != NEUTRAL_ID:
            enemies.append(neighbour)

    # Compute how much overkill damage the enemy would do if it would attack the `target` square
    for enemy in enemies:
        overkill_damage += min(enemy.strength, source.strength)

    # Return the normalized `overkill` damage
    overkill_factor_normalized = overkill_damage / MAX_STRENGTH
    return overkill_factor_normalized


def get_max_radar():
    """
    Computes a "radar" that represents the radius from the `square`
    """

    # Calculate the area that we currently own
    territory_size = 0
    for square in game_map:
        if square.owner == my_id:
            territory_size += 1
    
    # The radar consists of a weight determined by half the length of the map and how 
    # big our territory is. At first, when we have a small territory, we want to see
    # which are the best areas on the map
    return (WIDTH / 2) * math.exp(-territory_size / random.randint(900, 1100))


def check_valid_target(stack, targets, trgt, move, available_directions):
    """
    Computes the best moves for capturing a square `target`, updating the `stack` of moves and `targets` grid
    It uses a Greedy algorithm, sorting the combined moves in decreasing order by strength.
    If we can't put all the moves, we will put only the best ones (those which consume the least strength)
    @param stack:                   Current stack configuration of moves
    @param targets:                 Available targets
    @param trgt:                    Target to check if it's worth to capture it
    @param available_directions:    Valid directions to go through
    @return:                        Updated `stack` and `targets`
    """

    def get_total_strength_moves(combined_mvs):
        res = 0
        for move_dict in combined_mvs:
            res += move_dict['move'].square.strength
        return res

    # For each square, we build a list which combine strengths according to the rules of the 
    # game. It is worth joining the squares, but not exceeding the 255 limit.
    combined_mvs  = list(targets[trgt.y][trgt.x])
    combined_mvs += [{'move': move, 'avail_dirs': available_directions}]

    total_str = 0
    top_mvs   = []

    if get_total_strength_moves(combined_mvs) <= MAX_STRENGTH * random.uniform(1.1, 1.3):
        targets[trgt.y][trgt.x] = combined_mvs
    else:
        # Sort the combined moves decreasing by strength (we use a Greedy algorithm)

        # before sort: -1000 -50 -15 -1000 -200
        # after  sort: -1000 -1000 -200 -50 -15
        combined_mvs_idx = [
            -(MAX_STRENGTH + 1) 
                if mv['move'].direction == STILL
                else
            -mv['move'].square.strength for mv in combined_mvs]

        indices = [i for i in range(len(combined_mvs_idx))]
        indices.sort(key=lambda item: combined_mvs_idx[item])
        combined_mvs_sorted = [combined_mvs[i] for i in indices]
        
        for mv in combined_mvs_sorted:
            # We try to put all the movements, otherwise we put the best ones,
            # which consume the least strength, for this reason they are sorted in descending order.
            curr_str = mv['move'].square.strength
            if MAX_STRENGTH < total_str + curr_str:
                dirs = mv['avail_dirs']
                dirs.remove(mv['move'].direction)
                stack.append({
                    'sqr': mv['move'].square,
                    'avail_dirs': dirs
                    })
            else:
                # If we don't want to consider the square that would have the direction of our 
                # target, we put it in the stack again to look for a more optimal direction.
                # If it has been deposited during the current movement, it is removed from the directions
                total_str += curr_str
                top_mvs.append(mv)

        # Get the best moves for the square in position (x, y)
        targets[trgt.y][trgt.x] = top_mvs

    return stack, targets


def get_best_collective_moves():
    """
    Computes the best collective moves by processing a stack of individual moves,
    taking care of the self-destructive moves (e.g: moves which have combined_strength > MAX_STRENGTH)
    @return: List of moves which is send to the engine to make the moves for each square
    """

    # Create a dictionary that puts all directions (`stack`) for efficiency
    # Initializes each target field with None
    targets = [[None for _ in range(WIDTH)] for _ in range(HEIGHT)]
    
    # At first for each square we have all the directions available to "check"
    # This stack is used for storing the squares that need processing 
    stack = []
    for sq in game_map:
        if sq.owner == my_id:
            stack.append({'sqr': sq, 'avail_dirs': {NORTH, EAST, SOUTH, WEST}})

    # Go through the stack one by one
    while True:
        if len(stack) <= 0:
            break
        
        sqr         = stack[-1]['sqr']
        avail_dirs  = stack[-1]['avail_dirs']
        stack.pop()

        move        = compute_best_move_square(sqr, avail_dirs)
        trgt        = game_map.get_target(sqr, move.direction)

        if targets[trgt.y][trgt.x]:
            stack, targets = check_valid_target(stack, targets, trgt, move, avail_dirs)
        else:
            # If a squre is unvisited (None) it is marked that it can be reached and the directions to it
            targets[trgt.y][trgt.x] = [{'move': move, 'avail_dirs': avail_dirs}]

    # Complete the returned list with a move for each square
    return compute_final_moves(targets)


def compute_final_moves(targets):
    """
    Complete the returned list with a move for each square
    @param targets: matrix of possible moves
    @return:
    """
    
    moves = []
    for row in targets:
        for site in row:
            if site is not None:
                moves += [mv['move'] for mv in site]

    return moves


def count_enemy_bots():
    """
    Return the number of enemy bots, excluding our own and the neutral one
    """

    players_ids = []
    for square in game_map:
        players_ids.append(square.owner)

    return len(set(players_ids)) - 2


# Init the engine and get the initial `MAX_PRODUCTION` value from the map
my_id, game_map = hlt.get_init()
MAX_PRODUCTION = max([sq.production for sq in game_map])
hlt.send_init('TheEmpire')

# Get the dimensions for the play area
WIDTH  = game_map.width
HEIGHT = game_map.height

while True:
    # Get the current frame of the map
    game_map.get_frame()

    # Get the radar that represents the radius from the `square`
    RADAR           = int(get_max_radar())
    DIAGONAL_RADAR  = int(RADAR / 2)

    # Get the number of enemy bots
    number_of_enemy_bots = count_enemy_bots()

    # Get the moves for each square and send them to the server
    moves = get_best_collective_moves()
    hlt.send_frame(moves)

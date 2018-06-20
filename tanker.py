# noinspection PyPep8Naming
class Tanker:
    """The tanker class is used to move fuel from a ParentNode to a ChildNode, and
    is accomplished using type 'beta' tankers. In addition, tankers are used at a
    ChildNode to simulate demand for fuel using type 'gamma' tankers.  Each tanker
    has a unique name, tanker type ('beta', or 'gamma'), type of fuel it carries,
    amount of fuel currently on board, the node the tanker is coming from, the node
    the tanker is going to, and the TMR (if any) that the tanker is travelling under.
    @type name: str
    @param name: Name of the tanker, should be unique.
    @type tanker_type: str
    @param tanker_type: Type of tanker ('beta', 'gamma').
    @type capacity: float
    @param capacity: Total volume (gal) that can be transported by the tanker.
    @type fuel_type: str
    @param fuel_type: Type of fuel carried by the tanker.
    @type on_board: float
    @param on_board: Volume of fuel currently on board the tanker (gal).
    @type coming_from: parent_node.ParentNode | child_node.ChildNode
    @param coming_from: Node that the tanker is travelling from.
    @type going_to: parent_node.ParentNode | child_node.ChildNode
    @param going_to: Node that the tanker is travelling to.
    @type TMR_info: tmr.TMR
    @param TMR_info: Instance of TMR that identifies the TMR details.
    """
    def __init__(self, name, tanker_type, capacity, fuel_type, on_board=0.0,
                 coming_from=None, going_to=None, TMR_info=None):
        """The tanker class is used to move fuel from a ParentNode to a ChildNode, and
        is accomplished using type 'beta' tankers. In addition, tankers are used at a
        ChildNode to simulate demand for fuel using type 'gamma' tankers.  Each tanker
        has a unique name, tanker type ('beta', or 'gamma'), type of fuel it carries,
        amount of fuel currently on board, the node the tanker is coming from, the node
        the tanker is going to, and the TMR (if any) that the tanker is travelling under.
        @type name: str
        @param name: Name of the tanker, should be unique.
        @type tanker_type: str
        @param tanker_type: Type of tanker ('beta', 'gamma').
        @type capacity: float
        @param capacity: Total volume (gal) that can be transported by the tanker.
        @type fuel_type: str
        @param fuel_type: Type of fuel carried by the tanker.
        @type on_board: float
        @param on_board: Volume of fuel currently on board the tanker (gal).
        @type coming_from: parent_node.ParentNode | child_node.ChildNode
        @param coming_from: Node that the tanker is travelling from.
        @type going_to: parent_node.ParentNode | child_node.ChildNode
        @param going_to: Node that the tanker is travelling to.
        @type TMR_info: tmr.TMR
        @param TMR_info: Instance of TMR that identifies the TMR details.
        """
        self.log_label = name + '_' + tanker_type + '_' + fuel_type
        self.name = name
        self.tanker_type = tanker_type
        self.capacity = float(capacity)
        self.fuel_type = fuel_type
        self.on_board = float(on_board)
        self.coming_from = coming_from
        self.going_to = going_to
        self.TMR_info = TMR_info
        # SIM_STATS ('Time', 'Action', 'Loc_1', 'Loc_2', 'Fuel_Amount')
        self.status_trace = []  # SIM_STATS
        """@type : list[(float, str, str, str, float)]"""

    def __str__(self):
        """
        @rtype: str
        """
        if self.coming_from is None:
            return 'Tanker %s\n' \
                'Capacity: %.1f\tOn-board: %.1f\n' \
                'Fuel type: %s\n' \
                'Coming from: -\tGoing to: -\n' \
                % \
                (self.name, self.capacity, self.on_board, self.fuel_type)

        return 'Tanker %s\n' \
            'Capacity: %.1f\tOn-board: %.1f\n' \
            'Fuel type: %s\n' \
            'Coming from: %s\tGoing to: %s\n' \
            % \
            (self.name, self.capacity, self.on_board, self.fuel_type,
             self.coming_from.name, self.going_to.name)

    def update_on_board(self, scn, new_on_board):
        """Update the on_board amount.  The new_on_board value MUST be greater
        than 0 and less than the capacity of the tanker.  The checks in this
        function allow new_on_board to satisfy these constraints within the
        fuel_epsilon tolerance.
        @type scn: scenario.Scenario
        @type new_on_board: float
        """
        if not (-scn.fuel_epsilon <= new_on_board <= (self.capacity + scn.fuel_epsilon)):
            assert_msg = 'Tanker capacity constraints violated\nNew on_board: %.2f' % new_on_board
            assert False, assert_msg

        if abs(new_on_board) <= scn.fuel_epsilon:
            self.on_board = 0.0
        elif abs(self.capacity - new_on_board) <= scn.fuel_epsilon:
            self.on_board = self.capacity
        else:
            self.on_board = new_on_board

    def most_recent_loading_start_time(self):
        """Returns the most recent time the tanker began loading at the Parent node.
        @rtype : float
        """
        # Relevant tanker status_trace entries in reversed order
        # Tanker gets placed on LP:
        # tanker.status_trace.append((sim_time.time, 'on', self.name, self.node_name, fuel_amount))
        # Tanker gets placed on loading_Q:
        # tanker.status_trace.append((sim_time.time, 'on', 'loading_Q', FSP.name, np.nan))
        prev_entry = None
        for entry in reversed(self.status_trace):
            if (entry[1] == 'on') and (entry[2] == 'loading_Q'):
                return prev_entry[0]
            prev_entry = entry
        assert False, 'This tanker was never sent out from the Parent node:\n%s' % self.status_trace

    @staticmethod
    def load_time(LP, fuel_amount):
        """Return the loading time (in hours) of the tanker (a
        function of the (constant) loading point load rate [gal/hr]
        and the fuel_amount [gal] to be loaded).
        @type LP: lp.LoadingPoint
        @type fuel_amount: float
        @rtype: float
        """
        assert LP.upload_rate != 0, 'Trying to load a tanker on a LP that cannot upload fuel.'
        return fuel_amount / LP.upload_rate

    @staticmethod
    def unload_time(LP, fuel_amount):
        """Return the unloading time (in hours) of the tanker (a
        function of the (constant) loading point unload rate [gal/hr]
        and the fuel_amount [gal] to be unloaded).
        @type LP: lp.LoadingPoint
        @type fuel_amount: float
        @rtype: float
        """
        assert LP.download_rate != 0, 'Trying to unload a tanker on a LP that cannot download fuel.'
        return fuel_amount / LP.download_rate

library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

entity ternary_adder is
    generic (
        BW_INPUT0 : integer := 32;
        SIGNED0   : integer := 0;
        NEGATE0   : integer := 0;
        PAD0      : integer := 0;
        BW_INPUT1 : integer := 32;
        SIGNED1   : integer := 0;
        NEGATE1   : integer := 0;
        PAD1      : integer := 0;
        BW_INPUT2 : integer := 32;
        SIGNED2   : integer := 0;
        NEGATE2   : integer := 0;
        PAD2      : integer := 0;
        BW_OUT    : integer := 32;
        DROP_LSBS : integer := 0
    );
    port (
        in0    : in  std_logic_vector(BW_INPUT0-1 downto 0);
        in1    : in  std_logic_vector(BW_INPUT1-1 downto 0);
        in2    : in  std_logic_vector(BW_INPUT2-1 downto 0);
        result : out std_logic_vector(BW_OUT-1 downto 0)
    );
end entity ternary_adder;

architecture rtl of ternary_adder is
    function max2(lhs : integer; rhs : integer) return integer is
    begin
        if lhs > rhs then
            return lhs;
        else
            return rhs;
        end if;
    end function;

    function min2(lhs : integer; rhs : integer) return integer is
    begin
        if lhs < rhs then
            return lhs;
        else
            return rhs;
        end if;
    end function;

    function align_term(inp : std_logic_vector; is_signed : integer; right_pad : integer; out_bw : integer)
        return std_logic_vector is
        variable ret : std_logic_vector(out_bw-1 downto 0);
    begin
        if is_signed = 1 then
            ret := std_logic_vector(shift_left(resize(signed(inp), out_bw), right_pad));
        else
            ret := std_logic_vector(shift_left(resize(unsigned(inp), out_bw), right_pad));
        end if;
        return ret;
    end function;

    function select3(sel : integer; val0 : integer; val1 : integer; val2 : integer) return integer is
    begin
        if sel = 0 then
            return val0;
        elsif sel = 1 then
            return val1;
        else
            return val2;
        end if;
    end function;

    function bool_to_int(cond : boolean) return integer is
    begin
        if cond then
            return 1;
        else
            return 0;
        end if;
    end function;

    function first_min_input(pad0 : integer; pad1 : integer; min_pad : integer) return integer is
    begin
        if pad0 = min_pad then
            return 0;
        elsif pad1 = min_pad then
            return 1;
        else
            return 2;
        end if;
    end function;

    function next_pad(pad : integer; min_pad : integer; fallback : integer) return integer is
    begin
        if pad > min_pad then
            return pad;
        else
            return fallback;
        end if;
    end function;

    constant OUT_HIGH_ABS    : integer := DROP_LSBS + BW_OUT - 1;
    constant MIN_PAD         : integer := min2(min2(PAD0, PAD1), PAD2);
    constant LOW_COUNT       : integer := bool_to_int(PAD0 = MIN_PAD) + bool_to_int(PAD1 = MIN_PAD) + bool_to_int(PAD2 = MIN_PAD);
    constant LOW_INPUT       : integer := first_min_input(PAD0, PAD1, MIN_PAD);
    constant LOW_NEGATE      : integer := select3(LOW_INPUT, NEGATE0, NEGATE1, NEGATE2);
    constant NO_NEXT_PAD     : integer := OUT_HIGH_ABS + 2;
    constant NEXT_PAD0       : integer := next_pad(PAD0, MIN_PAD, NO_NEXT_PAD);
    constant NEXT_PAD1       : integer := next_pad(PAD1, MIN_PAD, NO_NEXT_PAD);
    constant NEXT_PAD2       : integer := next_pad(PAD2, MIN_PAD, NO_NEXT_PAD);
    constant UPPER_START     : integer := min2(min2(NEXT_PAD0, NEXT_PAD1), NEXT_PAD2);
    constant LOW_COPY_ABS_LO : integer := max2(DROP_LSBS, MIN_PAD);
    constant LOW_COPY_ABS_HI : integer := min2(OUT_HIGH_ABS, UPPER_START - 1);
    constant LOWCOPY_ENABLE  : boolean := LOW_COUNT = 1 and LOW_NEGATE = 0 and UPPER_START /= NO_NEXT_PAD
                                            and UPPER_START > DROP_LSBS and LOW_COPY_ABS_HI >= LOW_COPY_ABS_LO;
    constant FULL_MAX_TERM_BW: integer := max2(max2(BW_INPUT0 + PAD0, BW_INPUT1 + PAD1), BW_INPUT2 + PAD2);
    constant FULL_BW_ADD     : integer := max2(FULL_MAX_TERM_BW + 4, BW_OUT + DROP_LSBS + 1);
begin
    gen_lowcopy: if LOWCOPY_ENABLE generate
        constant LOW_PAD           : integer := select3(LOW_INPUT, PAD0, PAD1, PAD2);
        constant LOW_OUT_LO        : integer := LOW_COPY_ABS_LO - DROP_LSBS;
        constant LOW_OUT_HI        : integer := LOW_COPY_ABS_HI - DROP_LSBS;
        constant ZERO_HI           : integer := min2(OUT_HIGH_ABS, LOW_PAD - 1) - DROP_LSBS;
        constant SUFFIX_BW_RAW     : integer := OUT_HIGH_ABS - UPPER_START + 1;
        constant SUFFIX_BW         : integer := max2(SUFFIX_BW_RAW, 1);
        constant SUFFIX_MAX_TERM_BW: integer := max2(
            max2(max2(BW_INPUT0 + PAD0 - UPPER_START, 1), max2(BW_INPUT1 + PAD1 - UPPER_START, 1)),
            max2(BW_INPUT2 + PAD2 - UPPER_START, 1)
        );
        constant SUFFIX_BW_ADD     : integer := max2(SUFFIX_MAX_TERM_BW + 4, SUFFIX_BW + 1);
        constant EXT_BW            : integer := max2(DROP_LSBS + BW_OUT, UPPER_START + SUFFIX_BW_ADD);

        signal in0_ext : std_logic_vector(EXT_BW-1 downto 0);
        signal in1_ext : std_logic_vector(EXT_BW-1 downto 0);
        signal in2_ext : std_logic_vector(EXT_BW-1 downto 0);
        signal term0   : signed(SUFFIX_BW_ADD-1 downto 0);
        signal term1   : signed(SUFFIX_BW_ADD-1 downto 0);
        signal term2   : signed(SUFFIX_BW_ADD-1 downto 0);
        signal accum   : signed(SUFFIX_BW_ADD-1 downto 0);
    begin
        in0_ext <= align_term(in0, SIGNED0, PAD0, EXT_BW);
        in1_ext <= align_term(in1, SIGNED1, PAD1, EXT_BW);
        in2_ext <= align_term(in2, SIGNED2, PAD2, EXT_BW);

        term0 <= -signed(in0_ext(UPPER_START+SUFFIX_BW_ADD-1 downto UPPER_START))
                 when NEGATE0 = 1 else signed(in0_ext(UPPER_START+SUFFIX_BW_ADD-1 downto UPPER_START));
        term1 <= -signed(in1_ext(UPPER_START+SUFFIX_BW_ADD-1 downto UPPER_START))
                 when NEGATE1 = 1 else signed(in1_ext(UPPER_START+SUFFIX_BW_ADD-1 downto UPPER_START));
        term2 <= -signed(in2_ext(UPPER_START+SUFFIX_BW_ADD-1 downto UPPER_START))
                 when NEGATE2 = 1 else signed(in2_ext(UPPER_START+SUFFIX_BW_ADD-1 downto UPPER_START));
        accum <= term0 + term1 + term2;

        gen_zero: if LOW_PAD > DROP_LSBS generate
            result(ZERO_HI downto 0) <= (others => '0');
        end generate;

        gen_low0: if LOW_INPUT = 0 generate
            result(LOW_OUT_HI downto LOW_OUT_LO) <= in0_ext(LOW_COPY_ABS_HI downto LOW_COPY_ABS_LO);
        end generate;

        gen_low1: if LOW_INPUT = 1 generate
            result(LOW_OUT_HI downto LOW_OUT_LO) <= in1_ext(LOW_COPY_ABS_HI downto LOW_COPY_ABS_LO);
        end generate;

        gen_low2: if LOW_INPUT = 2 generate
            result(LOW_OUT_HI downto LOW_OUT_LO) <= in2_ext(LOW_COPY_ABS_HI downto LOW_COPY_ABS_LO);
        end generate;

        gen_upper: if UPPER_START <= OUT_HIGH_ABS generate
            result(BW_OUT-1 downto UPPER_START-DROP_LSBS) <= std_logic_vector(accum(SUFFIX_BW-1 downto 0));
        end generate;
    end generate;

    gen_full: if not LOWCOPY_ENABLE generate
        signal in0_ext : std_logic_vector(FULL_BW_ADD-1 downto 0);
        signal in1_ext : std_logic_vector(FULL_BW_ADD-1 downto 0);
        signal in2_ext : std_logic_vector(FULL_BW_ADD-1 downto 0);
        signal term0   : signed(FULL_BW_ADD-1 downto 0);
        signal term1   : signed(FULL_BW_ADD-1 downto 0);
        signal term2   : signed(FULL_BW_ADD-1 downto 0);
        signal accum   : signed(FULL_BW_ADD-1 downto 0);
    begin
        in0_ext <= align_term(in0, SIGNED0, PAD0, FULL_BW_ADD);
        in1_ext <= align_term(in1, SIGNED1, PAD1, FULL_BW_ADD);
        in2_ext <= align_term(in2, SIGNED2, PAD2, FULL_BW_ADD);

        term0 <= -signed(in0_ext) when NEGATE0 = 1 else signed(in0_ext);
        term1 <= -signed(in1_ext) when NEGATE1 = 1 else signed(in1_ext);
        term2 <= -signed(in2_ext) when NEGATE2 = 1 else signed(in2_ext);
        accum <= term0 + term1 + term2;

        result <= std_logic_vector(accum(BW_OUT-1+DROP_LSBS downto DROP_LSBS));
    end generate;
end architecture rtl;

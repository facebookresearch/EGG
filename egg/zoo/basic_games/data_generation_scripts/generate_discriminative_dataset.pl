#!/usr/bin/perl -w

#!/usr/bin/perl -w

# This script generates a data set in the format expected by EGG's basic_games discrimination game.

# Note that the script only generates distinct input tuples. More
# precisely, no input item *set* is ever repeated (e.g., if one input
# tuple contains (A, B, C), no other input line will be made of the
# same items, e.g., no (B, A, C) or (C, B, A) is permitted to occur in
# the same data set).

# Usage:

# perl -w N_INPUTS N_ATTRIBUTES N_VALUES N_ITEMS > discri_dataset.txt

# where

# N_INPUTS specifies the number of input tuples we are requesting,
# N_ATTRIBUTES specifies the number of attributes of each item,
# N_VALUES specifies the number of possible values for each attribute and
# N_ITEMS specifies how many items there are in a tuple (target+distractors).

# Note that, in the output, the requested tuples will be followed by a random index pointing to the position of the target (counting from 0).

# Note also that 0 is a possible value, thus, if N values are requested, the highest possible integer observed in the output will be N-1



# taken from: https://stackoverflow.com/questions/4736626/how-can-i-generate-all-ordered-combinations-of-length-k-in-perl
sub ordered_combinations
{
  my ($data, $k) = @_;

  return @$data if $k == 1;

  my @previous = ordered_combinations($data, $k-1);

  my @results;
  for my $symbol (@$data) {
    push @results, map { "$symbol " . $_ } @previous;
  }

  return @results;
} 

# shuffle( \@array ) : generate a random permutation
# of @array in place
# from https://www.oreilly.com/library/view/perl-cookbook/1565922433/ch04s18.html
sub shuffle {
    my $array = shift;
    my $i;
    for ($i = @$array; --$i; ) {
        my $j = int rand ($i+1);
        next if $i == $j;
        @$array[$i,$j] = @$array[$j,$i];
    }
}

# from: https://rosettacode.org/wiki/Evaluate_binomial_coefficients#Perl
sub binom {
    use bigint;
    my($n,$k) = @_;
    (0+$n)->bnok($k);
}


$n_distinct_samples = shift;
$item_length = shift;
$vocabulary_size = shift;
$n_items = shift;

# sanity check
$possible_distinct_items_count = $vocabulary_size**$item_length;
$max_samples_count = binom($possible_distinct_items_count,$n_items);
if ($n_distinct_samples>$max_samples_count) {
    print "with $n_items distinct items of length $item_length with $vocabulary_size values, I can maximally generate $max_samples_count samples\n";
    exit;
}

$vocabulary_size--;

@possible_items = (0..$vocabulary_size);
@all_distinct_combinations = ordered_combinations(\@possible_items,$item_length);

$current_sample_n = 0;
while ($current_sample_n<$n_distinct_samples) {
    @random_indices = (0..$#all_distinct_combinations);
    shuffle(\@random_indices);
    
    $sorted_selected_indices = join " ",(sort(@random_indices[0..$n_items-1]));
    if (!$seen_items{$sorted_selected_indices}) {
	@selected_items = @all_distinct_combinations[@random_indices[0..$n_items-1]];
	$target = int(rand($n_items));
	print join(" . ",(@selected_items,$target)),"\n";
	$seen_items{$sorted_selected_indices} = 1;
	$current_sample_n++;
    }
}



#!/usr/bin/perl -w

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
$sequence_length = shift;
$vocabulary_size = shift;
$n_items = shift;

# sanity check
$possible_distinct_items_count = $vocabulary_size**$sequence_length;
$max_samples_count = binom($possible_distinct_items_count,$n_items);
if ($n_distinct_samples>$max_samples_count) {
    print "with $n_items distinct items of length $sequence_length and vocabulary size $vocabulary_size, I can maximally generate $max_samples_count samples\n";
    exit;
}

$vocabulary_size--;

@possible_sequences = (0..$vocabulary_size);
@all_distinct_combinations = ordered_combinations(\@possible_sequences,$sequence_length);

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



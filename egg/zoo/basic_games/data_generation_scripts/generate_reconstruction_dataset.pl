#!/usr/bin/perl -w

# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# This script generates a data set in the format expected by EGG's basic_games reconstruction game.

# Note that the script only generates distinct items: you migt want to sample with replacement from
# its output to generate larger data sets with repeated items.

# Usage:

# perl -w N_INPUTS N_ATTRIBUTES N_VALUES > reco_dataset.txt

# where

# N_INPUTS specifies the number of items we are requesting,
# N_ATTRIBUTES specifies the number of attributes of each item and
# N_VALUES specifies the number of possible values for each attribute.

# Note that 0 is a possible value, thus if N values are requested the highest possible integer observed in the output will be N-1


$data_points = shift;
$attribute_n = shift;
$value_n = shift;

# sanity check
$possible_distinct_items_count = $value_n**$attribute_n;
if ($data_points>$possible_distinct_items_count) {
    print "I can maximally generate $possible_distinct_items_count distinct items of length $attribute_n with $value_n values\n";
    exit;
}


$n=0;
while ($n<$data_points) {
    $original_data_point = 0;
    while (!($original_data_point)) {
	@att_values = ();
	$curr_n = 0;
	while ($curr_n<$attribute_n) {
	    push @att_values,int(rand($value_n));
	    $curr_n++;
	}
	$data_point_as_string = join " ",@att_values;
	if (!$seen_data_points{$data_point_as_string}) {
	    $seen_data_points{$data_point_as_string}=1;
	    print join(" ",@att_values),"\n";
	    $original_data_point = 1;
	}
    }
    $n++;
}
